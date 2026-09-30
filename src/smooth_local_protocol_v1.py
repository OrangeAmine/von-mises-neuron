"""Source and evidence guards for the prospectively frozen VM-N01 study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> bytes:
    """Serialize finite JSON deterministically for evidence identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: Any) -> str:
    """Hash canonical JSON."""
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    """Hash a file incrementally without loading checkpoints into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    """Write finite reviewable evidence, refusing accidental overwrite."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contained_path(root: Path, relative: str) -> Path:
    """Resolve an evidence path while rejecting escapes and Git internals."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or ".git" in Path(relative).parts:
        raise ValueError(f"Uncontained evidence path: {relative}")
    return path


def inventory(paths: list[Path], root: Path = ROOT) -> dict[str, str]:
    """Return hashes indexed by contained repository-relative paths."""
    return {path.resolve().relative_to(root.resolve()).as_posix(): file_hash(path)
            for path in sorted(set(paths))}


def verify_inventory(entries: Mapping[str, str], root: Path = ROOT) -> None:
    """Reject any missing or altered source/evidence byte."""
    for relative, expected in entries.items():
        path = contained_path(root, relative)
        if file_hash(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {relative}")


def load_config(path: Path = ROOT / "configs/vm_n01_v1.json") -> dict[str, Any]:
    """Load the exact prospective configuration, checking its freeze receipt."""
    config = json.loads(path.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "docs/vm_n01_protocol_freeze.json").read_text())
    if file_hash(path) != freeze["files"]["configs/vm_n01_v1.json"]:
        raise ValueError("Prospective configuration changed after freeze")
    verify_inventory(freeze["files"])
    return config


def verify_source_lock(path: Path, config: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Verify source freeze and required pre-experiment check identities."""
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt["kind"] != "vm_n01_source_lock_v1" or receipt["config_sha256"] != digest(config):
        raise ValueError("Wrong source lock or configuration")
    verify_inventory(receipt["sources"], root)
    verify_inventory(receipt["checks"], root)
    return receipt


def verify_lock(path: str | Path, config: Mapping[str, Any], *, root: Path = ROOT,
                allow_preflight: bool = False) -> dict[str, Any]:
    """Verify all prerequisites before any reserved function is generated.

    Args:
        path: Confirmation lock created after complete pilot/development.
        config: Exact prospective configuration.
        root: Repository root, overridden only by isolated unit-test fixtures.
        allow_preflight: Permit a timing-only proof during lock construction.
            Reserved target generation never enables this exception.

    Returns:
        Verified receipt augmented with its actual file hash.
    """
    import math
    from .smooth_local_analysis_v1 import select_learning_rates, select_models

    lock_path = Path(path)
    receipt = json.loads(lock_path.read_text(encoding="utf-8"))
    if receipt.get("preflight_only", False) and not allow_preflight:
        raise ValueError("Preflight timing proofs do not authorize reserved access")
    if receipt["kind"] != "vm_n01_confirmation_lock_v1" or receipt["config_sha256"] != digest(config):
        raise ValueError("Wrong confirmation lock or configuration")
    if not all(receipt["prerequisites"].get(key) is True for key in
               ("full_tests", "standard_adapter_gate", "independent_review", "complete_development", "complete_pilot", "feasibility")):
        raise ValueError("Confirmation prerequisite is incomplete")
    required_sources = {
        "src/smooth_local_v1.py", "src/smooth_local_training_v1.py",
        "src/smooth_local_tasks_v1.py", "src/smooth_local_protocol_v1.py",
        "src/smooth_local_analysis_v1.py", "src/smooth_local_forecaster_v1.py",
        "src/locality_models.py", "src/circular_layers_v2.py",
        "scripts/run_vm_n01.py", "scripts/report_vm_n01.py", "scripts/validate_smooth_local_forecaster_v1.py",
        "scripts/benchmark_standard_dataset.py", "configs/vm_n01_v1.json",
        "docs/vm_n01_protocol.md", "docs/vm_n01_protocol_freeze.json",
    }
    if not required_sources.issubset(receipt.get("sources", {})):
        raise ValueError("Confirmation lock is missing core scientific sources")
    if not receipt.get("evidence") or not isinstance(receipt.get("source_lock_path"), str):
        raise ValueError("Confirmation lock requires evidence and a source lock reference")
    if set(receipt.get("phase_directories", {})) != {"pilot", "development"}:
        raise ValueError("Confirmation lock requires pilot and development phase references")
    verify_inventory(receipt["sources"], root)
    verify_inventory(receipt["evidence"], root)

    def bound(relative: str) -> Path:
        resolved = contained_path(root, relative)
        normalized = resolved.relative_to(root.resolve()).as_posix()
        if normalized not in receipt["evidence"] or file_hash(resolved) != receipt["evidence"][normalized]:
            raise ValueError(f"Required proof is not bound in confirmation evidence: {relative}")
        return resolved

    source_path = bound(receipt["source_lock_path"])
    source = verify_source_lock(source_path, config, root=root)
    if source["sources"] != receipt["sources"] or not source.get("checks"):
        raise ValueError("Source lock has different sources or no engineering checks")
    if any(receipt["evidence"].get(name) != value for name, value in source["checks"].items()):
        raise ValueError("Source engineering checks are missing from confirmation evidence")
    summaries = {}
    rates: dict[str, float] = {}
    for phase in ("pilot", "development"):
        directory = contained_path(root, receipt["phase_directories"][phase])
        prefix = directory.relative_to(root.resolve()).as_posix()
        plan, summary, terminal = [json.loads(bound(f"{prefix}/{name}").read_text(encoding="utf-8"))
                                   for name in ("plan.json", "summary.json", "receipt.json")]
        if (plan.get("phase") != phase or summary.get("phase") != phase
                or plan.get("config_sha256") != digest(config) or summary.get("config_sha256") != digest(config)
                or plan.get("source_lock_sha256") != file_hash(source_path)
                or summary.get("sources") != source["sources"]):
            raise ValueError("Prior phase has wrong stage, configuration or source identity")
        matrix = []
        labels = config["tasks"]["pilot_label_budgets" if phase == "pilot" else "label_budgets"]
        for family in config["tasks"]["families"]:
            for replicate in range(config["tasks"]["replicates"][phase]):
                for budget in labels:
                    for initialization in range(config["tasks"]["initializations"][phase]):
                        for spec in config["models"]:
                            choices = ([None] if spec["kind"] == "ridge" else
                                       config["training"]["pilot_learning_rates"] if phase == "pilot" else [rates[spec["id"]]])
                            for rate in choices:
                                matrix.append({"fit_id": f"fit_{len(matrix):05d}", "phase": phase,
                                               "family": family, "replicate": replicate, "labels": budget,
                                               "initialization": initialization, "model_id": spec["id"], "learning_rate": rate})
        count = config["phases"]["planned_fits"][phase]
        if len(matrix) != count or plan.get("matrix") != matrix:
            raise ValueError("Prior phase plan differs from the exact prospective matrix")
        elapsed = terminal.get("elapsed_seconds")
        ceiling = config["phases"]["ceiling_seconds"][phase]
        if (summary.get("complete") is not True or summary.get("interrupted") is not None
                or terminal.get("complete") is not True or terminal.get("within_ceiling") is not True
                or not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
                or not math.isfinite(elapsed) or not 0 <= elapsed <= ceiling
                or terminal.get("ceiling_seconds") != ceiling
                or terminal.get("summary_sha256") != file_hash(directory / "summary.json")
                or summary.get("planned_count") != count or summary.get("attempted_count") != count):
            raise ValueError("Prior phase terminal proof is incomplete, stale or over budget")
        if {p.name for p in directory.glob("fit_*.json")} != {f"{row['fit_id']}.json" for row in matrix}:
            raise ValueError("Prior phase fit records are missing or extra")
        rows = []
        for item in matrix:
            compact = json.loads(bound(f"{prefix}/{item['fit_id']}.json").read_text(encoding="utf-8"))
            artifact = bound(f"{prefix}/{item['fit_id']}.pt")
            if any(compact.get(key) != value for key, value in item.items()) or compact.get("artifact_sha256") != file_hash(artifact):
                raise ValueError("Prior fit identity or artifact identity differs from its proof")
            allowances = config["training"]["time_budgets_seconds"]
            if len(compact.get("rows", [])) != len(allowances):
                raise ValueError("Prior fit is missing allowance rows")
            for row, allowance in zip(compact["rows"], allowances):
                if (any(row.get(key) != value for key, value in item.items()) or row.get("allowance") != allowance
                        or row.get("status") != "completed"):
                    raise ValueError("Prior fit has incomplete or incorrect metric identities")
                for key in ("normalized_mse", "noisy_validation_mse", "selected_seconds"):
                    value = row.get(key)
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                        raise ValueError("Prior fit has invalid metrics or checkpoint timing")
                if row["selected_seconds"] > allowance:
                    raise ValueError("Prior fit selected a late checkpoint")
            rows.extend(compact["rows"])
        if rows != summary.get("rows"):
            raise ValueError("Prior phase summary differs from immutable fit records")
        if phase == "pilot":
            rates = select_learning_rates(rows, config)
        if summary.get("selected_rates") != rates:
            raise ValueError("Prior phase learning rates disagree with replayed selection")
        if phase == "development" and summary.get("selected_models") != select_models(rows, config):
            raise ValueError("Prior phase model choices disagree with replayed selection")
        summaries[phase] = summary
    if receipt.get("selected_rates") != rates or receipt.get("selected_models") != summaries["development"]["selected_models"]:
        raise ValueError("Confirmation choices disagree with complete prior evidence")
    feasibility = receipt["feasibility"]
    projected = feasibility.get("projected_seconds")
    if (feasibility.get("phase") != "confirmation" or feasibility.get("passed") is not True
            or not isinstance(projected, (int, float)) or isinstance(projected, bool)
            or not math.isfinite(projected) or projected < 0
            or feasibility.get("ceiling_seconds") != config["phases"]["ceiling_seconds"]["confirmation"]
            or projected > config["phases"]["ceiling_seconds"]["confirmation"]):
        raise ValueError("Confirmation exceeds the unchanged prospective ceiling")
    return dict(receipt, lock_sha256=file_hash(lock_path))


def scientific_paths(root: Path = ROOT) -> list[Path]:
    """Bind all executable study sources and reused equations, without v1/v2 edits."""
    paths = list((root / "src").glob("*.py")) + list((root / "tests").glob("*.py"))
    paths += list((root / "scripts").glob("*smooth_local*.py"))
    paths += [root / "scripts/run_vm_n01.py", root / "scripts/report_vm_n01.py", root / "scripts/benchmark_standard_dataset.py",
              root / "configs/vm_n01_v1.json", root / "docs/vm_n01_protocol.md",
              root / "docs/vm_n01_protocol_freeze.json", root / "docs/vm_n01_inherited_baseline.json"]
    return paths
