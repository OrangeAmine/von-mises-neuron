"""Versioned run provenance and procedural confirmation access controls.

The holdout is procedural, not cryptographically secret. A detached checksum
detects accidental lock edits; source and result hashes bind its dependencies.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vm_methodology_v1.json"
SOURCE_PATHS = [
    "src/circular_layers_v2.py", "src/methodology_forecasters.py",
    "src/methodology_tasks.py", "src/methodology_training.py",
    "src/methodology_protocol.py", "scripts/run_vm_methodology.py",
    "scripts/analyze_vm_methodology.py", "scripts/benchmark_circular_layers_v2.py",
    "src/von_mises_neuron.py", "src/temporal.py", "src/sincos.py",
    "docs/vm_methodology_plan.md", "configs/vm_methodology_v1.json",
]


def json_bytes(value: Any) -> bytes:
    """Encode strict canonical JSON for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def object_hash(value: Any) -> str:
    """Hash a canonical JSON object."""
    return hashlib.sha256(json_bytes(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash exact bytes, including uncommitted source changes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    """Read JSON, refusing nonstandard nonfinite constants."""
    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON constant {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    """Write finite JSON; mutable progress files use atomic replacement."""
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)


def environment() -> dict[str, Any]:
    """Record hardware, libraries and numerical determinism settings."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                             text=True).strip(),
    }


def source_manifest(repo_root: Path = ROOT) -> dict[str, str]:
    """Require all experiment implementation files and hash their exact bytes."""
    return {name: file_hash(repo_root / name) for name in SOURCE_PATHS}


def snapshot_sources(output: Path, repo_root: Path = ROOT) -> dict[str, str]:
    """Save independent source copies alongside a run, without overwrites."""
    manifest = source_manifest(repo_root)
    for name in manifest:
        destination = output / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write((repo_root / name).read_bytes())
    return manifest


def _checked_dependencies(mapping: dict[str, str], repo_root: Path) -> None:
    """Verify named dependency bytes stay within the study repository."""
    if not mapping:
        raise ValueError("Empty lock dependency manifest")
    for name, expected in mapping.items():
        path = (repo_root / name).resolve()
        if not path.is_relative_to(repo_root.resolve()):
            raise ValueError("Lock dependency outside repository")
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Changed or missing lock dependency: {name}")


def verify_confirmation_lock(lock_path: Path | None, config: dict[str, Any],
                             *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Reject missing, changed, unresolved or mismatched confirmation locks.

    Args:
        lock_path: Previously frozen JSON artifact, with .sha256 sidecar.
        config: The full canonical experiment configuration.
        repo_root: Repository containing exact source and development bytes.

    Returns:
        Verified lock contents with a computed lock_sha256 field.
    """
    if lock_path is None:
        raise ValueError("Confirmation requires a frozen lock")
    lock_path = Path(lock_path)
    sidecar = lock_path.with_name(lock_path.name + ".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise ValueError("Confirmation lock or detached checksum is missing")
    digest = file_hash(lock_path)
    if sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("Confirmation lock changed")
    lock = read_json(lock_path)
    required = config["execution"]["confirmation_requires_lock"]
    if any(name not in lock for name in required):
        raise ValueError("Incomplete confirmation lock")
    if lock["config_hash"] != object_hash(config):
        raise ValueError("Confirmation configuration changed")
    if lock["analysis_rules"] != config["analysis"]:
        raise ValueError("Confirmation analysis rules changed")
    if lock["B_primary_comparator"] not in config["B"]["proper_comparators"]:
        raise ValueError("Invalid primary comparator")
    choices = lock["pilot_hyperparameters"]
    for optimizer, spec in config["A"]["optimizers"].items():
        if choices.get("A", {}).get(optimizer) not in spec["pilot_learning_rates"]:
            raise ValueError("Unresolved A learning rate")
    for spec in config["B"]["models"]:
        if spec["kind"] == "gradient" and choices.get("B", {}).get(spec["id"]) not in config["B"]["optimizer"]["pilot_learning_rates"]:
            raise ValueError("Unresolved B learning rate")
    for key in ("source_hashes", "generator_hashes", "development_results_hashes"):
        _checked_dependencies(lock[key], repo_root)
    if set(lock["source_hashes"]) != set(SOURCE_PATHS):
        raise ValueError("Incomplete source manifest")
    if lock["generator_hashes"] != {"src/methodology_tasks.py": lock["source_hashes"]["src/methodology_tasks.py"]}:
        raise ValueError("Generator manifest mismatch")
    lock["lock_sha256"] = digest
    return lock


def write_confirmation_lock(path: Path, config: dict[str, Any],
                            pilot_choices: dict[str, Any], comparator: str,
                            dependencies: list[Path], *, repo_root: Path = ROOT) -> dict[str, Any]:
    """Freeze exact source/results/choices before reserved task generation."""
    hashes = source_manifest(repo_root)
    expanded = set(dependencies)
    for item in dependencies:
        if item.name != "manifest.json":
            continue
        evidence = read_json(item)
        for row in evidence.get("runs", []):
            expanded.update(item.parent/name for name in row.get("artifact_hashes", {}))
        for key, names in evidence.get("tasks", {}).items():
            expanded.update(item.parent/"tasks"/key/name for name in names)
    lock = {
        "protocol_id": config["protocol_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": object_hash(config), "source_hashes": hashes,
        "generator_hashes": {"src/methodology_tasks.py": hashes["src/methodology_tasks.py"]},
        "pilot_hyperparameters": pilot_choices,
        "development_results_hashes": {str(item.resolve().relative_to(repo_root.resolve())).replace("\\", "/"): file_hash(item) for item in sorted(expanded)},
        "B_primary_comparator": comparator, "analysis_rules": config["analysis"],
    }
    write_json(path, lock)
    with path.with_name(path.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(file_hash(path) + "\n")
    return verify_confirmation_lock(path, config, repo_root=repo_root)


def planned_runs(config: dict[str, Any], stage: str, phase: str,
                 choices: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Enumerate every gradient/ridge/reference cell before generating data."""
    if stage not in ("A", "B") or phase not in ("pilot", "development", "confirmation"):
        raise ValueError("Unknown stage or phase")
    rows: list[dict[str, Any]] = []
    spec = config[stage]
    families = spec["pilot"]["conditions"] if stage == "A" and phase == "pilot" else list(spec["family_ids"])
    for family in families:
        for replicate in range(spec[phase]["task_pairs"]):
            common = {"stage": stage, "phase": phase, "family": family, "replicate": replicate}
            if stage == "A":
                for penalty in spec["explicit_amplitude_penalties"]:
                    rows.append({**common, "model": "reference", "kind": "reference", "penalty": penalty})
                    for optimizer, opt in spec["optimizers"].items():
                        rates = opt["pilot_learning_rates"] if phase == "pilot" else [choices["A"][optimizer]]
                        for rate in rates:
                            for model in spec["models"]:
                                rows.append({**common, "model": model, "kind": "gradient", "optimizer": optimizer, "learning_rate": rate, "penalty": penalty})
            else:
                sizes = [spec["pilot"]["train_observations"]] if phase == "pilot" else spec["train_observations"]
                for size in sizes:
                    for model in spec["models"]:
                        if model["kind"] == "ridge":
                            rows.append({**common, "model": model["id"], "kind": "ridge", "n_train": size, "initialization_index": None})
                        else:
                            rates = spec["optimizer"]["pilot_learning_rates"] if phase == "pilot" else [choices["B"][model["id"]]]
                            for index in range(len(config["seeds"][phase]["initialization"])):
                                for rate in rates:
                                    rows.append({**common, "model": model["id"], "kind": "gradient", "n_train": size, "initialization_index": index, "learning_rate": rate})
    for index, row in enumerate(rows):
        row.update(run_id=f"{stage}_{phase}_{index:04d}", status="pending")
    return rows
