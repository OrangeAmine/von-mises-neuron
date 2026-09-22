"""Run the prospective, source-bound VM-N02 interpretation study.

Stages are immutable: failed or incomplete stages cannot be retried in place.
The reserved pool is generated only after source and precursor verification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "configs/vm_n02_interpretation_v1.json"
DEFAULT_PROTOCOL = ROOT / "docs/vm_n02_interpretation_protocol.md"
RUNNER_SOURCE = ROOT / "scripts/run_vm_n02_interpretation.py"
POSTFLIGHT_ERROR_PREFIX = "Observed competing GPU resident activity during timed stage: "
AMENDMENT_SCOPE = "postflight_cpu_activity_is_recorded_not_a_scientific_failure"
SCIENTIFIC_SOURCES = (
    ROOT / "scripts/run_vm_n02_interpretation.py",
    ROOT / "src/interpretation_n02_cases.py",
    ROOT / "src/interpretation_n02_diagnostics.py",
    ROOT / "src/circular_layers_v2.py",
    ROOT / "src/locality_models.py",
)
STAGES = ("pilot", "development", "evaluation")


def sha256(path: Path) -> str:
    """Hash a file without loading large checkpoints in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Preserve undefined numeric outcomes as JSON null, never silent NaNs."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: Any) -> None:
    """Write complete JSON using a same-directory atomic replacement."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _command(arguments: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            arguments, cwd=ROOT, capture_output=True, text=True, timeout=10,
            check=False,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def _process_cpu_snapshot(pid: int) -> dict[str, Any]:
    """Read a Windows process clock without attaching to or modifying it."""
    if os.name != "nt":
        return {"error": "Windows process-clock inspection unavailable on this platform"}
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION.
    if not handle:
        return {"error": f"OpenProcess failed: Windows error {ctypes.get_last_error()}"}
    try:
        created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(item) for item in (created, exited, kernel_time, user_time))):
            return {"error": f"GetProcessTimes failed: Windows error {ctypes.get_last_error()}"}
        ticks = lambda value: (value.dwHighDateTime << 32) + value.dwLowDateTime
        return {"cpu_seconds": (ticks(kernel_time) + ticks(user_time)) / 10_000_000,
                "created_ticks": ticks(created)}
    finally:
        kernel.CloseHandle(handle)


def _gpu_processes() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = _command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                      "--format=csv,noheader"])
    processes = []
    if query.get("returncode") == 0:
        for row in csv.reader(query["stdout"].splitlines()):
            if len(row) >= 2 and row[0].strip().isdigit():
                pid = int(row[0].strip())
                name = row[1].strip()
                base = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
                ignored = "this runner" if pid == os.getpid() else (
                    "known PowerToys desktop launcher" if base == "powertoys.powerlauncher.exe" else None)
                processes.append({"pid": pid, "name": name, "ignored_reason": ignored,
                                  "gpu_memory": row[2].strip() if len(row) >= 3 else None})
    return query, processes


def gpu_preflight(device: torch.device) -> dict[str, Any]:
    """Distinguish observed process activity from mere CUDA memory residency."""
    if device.type != "cuda":
        return {"gpu_process_query": {"not_requested": "CPU device"},
                "gpu_processes": [], "competing_gpu_processes": [], "gpu_cpu_check": {}}
    query, processes = _gpu_processes()
    candidates = [item for item in processes if not item["ignored_reason"]]
    before = {str(item["pid"]): _process_cpu_snapshot(item["pid"]) for item in candidates}
    checked_at = time.perf_counter()
    if candidates:
        time.sleep(5.0)
    after = {str(item["pid"]): _process_cpu_snapshot(item["pid"]) for item in candidates}
    competitors = []
    for item in candidates:
        first, second = before[str(item["pid"])], after[str(item["pid"])]
        unknown = "error" in first or "error" in second
        changed = (not unknown and (first["created_ticks"] != second["created_ticks"]
                                   or second["cpu_seconds"] - first["cpu_seconds"] > 1e-6))
        if unknown or changed:
            competitors.append({**item, "reason": "CPU clock advanced" if changed else "CPU clock unavailable"})
    return {"gpu_process_query": query, "gpu_processes": processes,
            "competing_gpu_processes": competitors,
            "gpu_cpu_check": {"before": before, "after": after,
                              "sample_seconds": time.perf_counter() - checked_at,
                              "limitation": "Five-second CPU quiescence distinguishes idle residency only; it cannot prove GPU idleness or exclude future activity."}}


def gpu_end_snapshot(device: torch.device, initial_environment: dict[str, Any]) -> dict[str, Any]:
    """Check whether other observed GPU residents progressed during this stage."""
    if device.type != "cuda":
        return {"gpu_processes": [], "competing_gpu_processes": []}
    query, processes = _gpu_processes()
    previous = initial_environment.get("gpu_cpu_check", {}).get("after", {})
    after = {str(item["pid"]): _process_cpu_snapshot(item["pid"])
             for item in processes if not item["ignored_reason"]}
    competitors = []
    for item in processes:
        if item["ignored_reason"]:
            continue
        first, second = previous.get(str(item["pid"])), after[str(item["pid"])]
        unknown = first is None or "error" in first or "error" in second
        changed = (not unknown and (first["created_ticks"] != second["created_ticks"]
                                   or second["cpu_seconds"] - first["cpu_seconds"] > 1e-6))
        if unknown or changed:
            competitors.append({**item, "reason": "CPU clock advanced during stage" if changed else "New or uninspectable GPU resident"})
    return {"gpu_process_query": query, "gpu_processes": processes,
            "cpu_after": after, "competing_gpu_processes": competitors,
            "limitation": "Start/end clocks cannot observe processes that both start and exit between snapshots or prove absence of GPU-only activity."}


def environment(device: torch.device) -> dict[str, Any]:
    """Record software, hardware, revision and observable GPU contention."""
    gpu = gpu_preflight(device)
    return {
        "utc": _utc(), "pid": os.getpid(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device": str(device), "dtype": "float64", "cpu_threads": torch.get_num_threads(),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "git_revision": _command(["git", "rev-parse", "HEAD"]),
        "git_status": _command(["git", "status", "--short"]),
        **gpu,
    }


def freeze_run(output_dir: Path, config_path: Path, protocol_path: Path,
               source_paths: tuple[Path, ...] | None = None) -> dict[str, Any]:
    """Freeze prospective configuration, protocol and scientific source hashes."""
    output_dir = output_dir.resolve()
    frozen = output_dir / "frozen"
    if frozen.exists() or any((output_dir / name).exists() for name in (*STAGES, "evaluation_lock.json")):
        raise FileExistsError(f"Run already frozen or used: {output_dir}")
    sources = list(SCIENTIFIC_SOURCES if source_paths is None else source_paths)
    sources.extend([config_path.resolve(), protocol_path.resolve()])
    records = []
    for path in dict.fromkeys(path.resolve() for path in sources):
        records.append({"path": str(path), "sha256": sha256(path)})
    config = read_json(config_path)
    _validate_config(config)
    frozen.mkdir(parents=True)
    shutil.copyfile(config_path, frozen / "config.json")
    shutil.copyfile(protocol_path, frozen / "protocol.md")
    manifest = {"version": config["version"], "created_utc": _utc(), "sources": records,
                "config_sha256": sha256(frozen / "config.json"),
                "protocol_sha256": sha256(frozen / "protocol.md"),
                "git_revision": _command(["git", "rev-parse", "HEAD"])}
    write_json(frozen / "source_manifest.json", manifest)
    return manifest


def _validate_config(config: dict[str, Any]) -> None:
    if set(config["pools"]) != set(STAGES):
        raise ValueError("Config must define pilot, development and evaluation pools")
    if config["cases"] != ["isolated", "redundant", "proxy"]:
        raise ValueError("The complete three-case matrix is required")
    if len(set(config["initializations"])) != len(config["initializations"]) or not config["initializations"]:
        raise ValueError("Initializations must be nonempty and unique")
    offsets = [config["pools"][stage]["seed_offset"] for stage in STAGES]
    if len(set(offsets)) != len(offsets):
        raise ValueError("Pool seed namespaces must be disjoint")
    for stage in STAGES:
        pool = config["pools"][stage]
        if not isinstance(pool["units"], int) or pool["units"] < 1 or pool["cap_seconds"] <= 0:
            raise ValueError(f"Invalid pool size or cap for {stage}")
    if config["compute"]["dtype"] != "float64":
        raise ValueError("This protocol requires float64")


def verify_freeze(output_dir: Path) -> tuple[dict[str, Any], str]:
    """Refuse source/config/protocol drift before any pool generation."""
    frozen = output_dir / "frozen"
    manifest_path = frozen / "source_manifest.json"
    manifest = read_json(manifest_path)
    amendment = _verify_source_amendment(output_dir, manifest)
    for record in manifest["sources"]:
        path = Path(record["path"])
        expected = (amendment["new_runner_sha256"] if amendment and path.resolve() == RUNNER_SOURCE.resolve()
                    else record["sha256"])
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen scientific source changed or missing: {path}")
    for name, key in (("config.json", "config_sha256"), ("protocol.md", "protocol_sha256")):
        if sha256(frozen / name) != manifest[key]:
            raise ValueError(f"Frozen copy changed: {name}")
    config = read_json(frozen / "config.json")
    _validate_config(config)
    return config, sha256(manifest_path)


def source_amendment_hash(output_dir: Path) -> str | None:
    """Return the operational amendment identity, if one was explicitly made."""
    path = output_dir / "source_amendment.json"
    return sha256(path) if path.is_file() else None


def _verify_source_amendment(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    path = output_dir / "source_amendment.json"
    if not path.exists():
        return None
    amendment = read_json(path)
    records = {str(Path(item["path"]).resolve()): item["sha256"] for item in manifest["sources"]}
    runner_path = str(RUNNER_SOURCE.resolve())
    archive = output_dir / "frozen/runner_before_postflight_fix.py"
    unchanged = {key: value for key, value in records.items() if key != runner_path}
    if (amendment.get("schema") != "vm_n02_operational_amendment_v1"
            or amendment.get("scope") != AMENDMENT_SCOPE
            or amendment.get("runner_path") != runner_path
            or amendment.get("source_manifest_sha256") != sha256(output_dir / "frozen/source_manifest.json")
            or amendment.get("old_runner_sha256") != records.get(runner_path)
            or amendment.get("archived_runner_relative_path") != "frozen/runner_before_postflight_fix.py"
            or amendment.get("unchanged_source_hashes") != unchanged
            or not archive.is_file()
            or sha256(archive) != amendment.get("old_runner_sha256")
            or amendment.get("archived_runner_sha256") != amendment.get("old_runner_sha256")
            or sha256(RUNNER_SOURCE) != amendment.get("new_runner_sha256")):
        raise ValueError("Invalid or changed narrowly scoped operational source amendment")
    return amendment


def _postflight_error(receipt: dict[str, Any]) -> None:
    if (receipt.get("status") != "error" or receipt.get("error_type") != "RuntimeError"
            or not receipt.get("error", "").startswith(POSTFLIGHT_ERROR_PREFIX)
            or not receipt.get("environment_end", {}).get("competing_gpu_processes")):
        raise ValueError("Only the exact completed-stage postflight CPU-activity RuntimeError is adjudicable")


def prepare_source_amendment(output_dir: Path, reason: str) -> dict[str, Any]:
    """Attest one operational runner correction without changing scientific bytes.

    The original runner must already be archived at the fixed path. This helper
    performs no data generation, fitting, metric selection or result mutation.
    """
    path = output_dir / "source_amendment.json"
    if path.exists() or any((output_dir / item).exists() for item in ("development", "evaluation", "evaluation_lock.json")):
        raise FileExistsError("Operational amendment must precede development and reserved access and cannot be replaced")
    _postflight_error(read_json(output_dir / "pilot/receipt.json"))
    manifest_path = output_dir / "frozen/source_manifest.json"
    manifest = read_json(manifest_path)
    runner_path = str(RUNNER_SOURCE.resolve())
    records = {str(Path(item["path"]).resolve()): item["sha256"] for item in manifest["sources"]}
    if runner_path not in records:
        raise ValueError("Original freeze does not contain the canonical runner")
    for source, digest in records.items():
        if source != runner_path and sha256(Path(source)) != digest:
            raise ValueError(f"Scientific source changed; operational amendment forbidden: {source}")
    archive = output_dir / "frozen/runner_before_postflight_fix.py"
    if sha256(archive) != records[runner_path]:
        raise ValueError("Archived original runner does not match original source freeze")
    for name, key in (("config.json", "config_sha256"), ("protocol.md", "protocol_sha256")):
        if sha256(output_dir / "frozen" / name) != manifest[key]:
            raise ValueError(f"Frozen scientific copy changed: {name}")
    amendment = {"schema": "vm_n02_operational_amendment_v1", "scope": AMENDMENT_SCOPE,
                 "created_utc": _utc(), "reason": reason, "runner_path": runner_path,
                 "source_manifest_sha256": sha256(manifest_path),
                 "old_runner_sha256": records[runner_path], "new_runner_sha256": sha256(RUNNER_SOURCE),
                 "archived_runner_relative_path": "frozen/runner_before_postflight_fix.py",
                 "archived_runner_sha256": sha256(archive),
                 "unchanged_source_hashes": {key: value for key, value in records.items() if key != runner_path}}
    write_json(path, amendment)
    verify_freeze(output_dir)
    return amendment


def _inventory(directory: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.relative_to(directory).as_posix() not in excluded}


def verify_stage(output_dir: Path, stage: str, config: dict[str, Any],
                 source_hash: str) -> dict[str, Any]:
    """Verify completion, full matrix, source binding and every saved artifact."""
    directory = output_dir / stage
    original = read_json(directory / "receipt.json")
    adjudication_path = directory / "completion_adjudication.json"
    adjudicated = adjudication_path.is_file()
    receipt = read_json(adjudication_path) if adjudicated else original
    if adjudicated:
        _postflight_error(original)
        if (stage != "pilot" or receipt.get("status") != "complete_with_operational_adjudication"
                or receipt.get("original_receipt_sha256") != sha256(directory / "receipt.json")
                or receipt.get("original_elapsed_seconds") != original["elapsed_seconds"]
                or receipt.get("elapsed_seconds", -1) < original["elapsed_seconds"]
                or receipt.get("completed_units") != original["completed_units"]
                or not source_amendment_hash(output_dir)):
            raise ValueError("Invalid completion adjudication or changed original error receipt")
    elif receipt["status"] != "complete":
        raise ValueError(f"{stage} was not completed; retries are forbidden")
    if receipt.get("source_amendment_sha256") != source_amendment_hash(output_dir):
        raise ValueError(f"{stage} does not bind the current operational source amendment")
    if receipt["source_manifest_sha256"] != source_hash:
        raise ValueError(f"{stage} belongs to a different source freeze")
    expected = {f"{case}_{index:03d}" for case in config["cases"]
                for index in range(config["pools"][stage]["units"])}
    if adjudicated:
        prior_units = receipt.get("unit_artifacts_before_adjudication", {})
        if set(prior_units) != expected or any(
                _inventory(directory / unit_id) != prior_units[unit_id] for unit_id in expected):
            raise ValueError("Adjudication does not preserve the exact original unit artifact inventory")
    if set(receipt["completed_units"]) != expected or len(receipt["completed_units"]) != len(expected):
        raise ValueError(f"{stage} is missing declared units or has duplicate units")
    cap = config["pools"][stage]["cap_seconds"]
    if receipt["elapsed_seconds"] > cap or receipt["elapsed_seconds"] < 0:
        raise ValueError(f"{stage} exceeded its fixed cap or has invalid timing")
    manifest = read_json(directory / "artifact_manifest.json")
    actual = _inventory(directory, {"artifact_manifest.json", "receipt.json", "completion_adjudication.json"})
    if actual != manifest["artifacts"]:
        raise ValueError(f"{stage} artifact hashes or inventory changed")
    if sha256(directory / "artifact_manifest.json") != receipt["artifact_manifest_sha256"]:
        raise ValueError(f"{stage} artifact manifest changed")
    for unit_id in expected:
        unit_receipt = read_json(directory / unit_id / "receipt.json")
        expected_fits = 1 if unit_id.startswith("proxy_") else len(config["initializations"])
        if unit_receipt["status"] != "complete" or unit_receipt["fits"] != expected_fits:
            raise ValueError(f"Incomplete or incorrect initialization count for {unit_id}")
    return receipt


def adjudicate_completed_postflight(output_dir: Path) -> dict[str, Any]:
    """Complete missing pilot receipts after the narrow recorded monitor error.

    Original error receipt and every unit file remain byte-identical. The
    separate terminal record charges measured finalization overhead and remains
    distinguishable from a clean run. No scientific API is loaded or called.
    """
    started = time.perf_counter()
    if any((output_dir / item).exists() for item in ("development", "evaluation", "evaluation_lock.json")):
        raise FileExistsError("Pilot adjudication must precede subsequent stages")
    config, source_hash = verify_freeze(output_dir)
    amendment_hash = source_amendment_hash(output_dir)
    if amendment_hash is None:
        raise ValueError("Explicit verified operational source amendment required")
    directory = output_dir / "pilot"
    for name in ("completion_adjudication.json", "stage_summary.json", "artifact_manifest.json"):
        if (directory / name).exists():
            raise FileExistsError(f"Adjudication only finalizes missing artifacts, without overwrite: {name}")
    original_path = directory / "receipt.json"
    original_hash = sha256(original_path)
    original = read_json(original_path)
    _postflight_error(original)
    expected = {f"{case}_{index:03d}" for case in config["cases"]
                for index in range(config["pools"]["pilot"]["units"])}
    if (original.get("stage") != "pilot" or original.get("source_manifest_sha256") != source_hash
            or set(original["completed_units"]) != expected or len(original["completed_units"]) != len(expected)):
        raise ValueError("Postflight error did not retain the exact complete pilot matrix and original source")
    cap = config["pools"]["pilot"]["cap_seconds"]
    if not 0 <= original["elapsed_seconds"] < cap:
        raise ValueError("Original pilot does not leave room in the unchanged compute cap")
    unit_hashes = {}
    for unit_id in sorted(expected):
        unit_dir = directory / unit_id
        unit_receipt = read_json(unit_dir / "receipt.json")
        fits = 1 if unit_id.startswith("proxy_") else len(config["initializations"])
        required = {"payload.pt", "metrics.json", "details.json", "arrays.npz", "receipt.json"}
        if (unit_receipt.get("status") != "complete" or unit_receipt.get("fits") != fits
                or not required.issubset({item.name for item in unit_dir.iterdir() if item.is_file()})):
            raise ValueError(f"Missing completed unit artifacts or fits: {unit_id}")
        unit_hashes[unit_id] = _inventory(unit_dir)
    write_json(directory / "stage_summary.json", {
        "stage": "pilot", "units_per_case": config["pools"]["pilot"]["units"],
        "completed_units": original["completed_units"], "initializations": config["initializations"],
        "source_manifest_sha256": source_hash, "source_amendment_sha256": amendment_hash,
        "operational_status": "Completed numerical artifacts; original postflight monitor error retained and separately adjudicated"})
    write_json(directory / "artifact_manifest.json", {
        "artifacts": _inventory(directory, {"receipt.json", "artifact_manifest.json", "completion_adjudication.json"})})
    adjudication = {**original, "status": "complete_with_operational_adjudication", "original_status": "error",
                    "original_receipt_sha256": original_hash, "original_elapsed_seconds": original["elapsed_seconds"],
                    "source_amendment_sha256": amendment_hash, "unit_artifacts_before_adjudication": unit_hashes,
                    "artifact_manifest_sha256": sha256(directory / "artifact_manifest.json"),
                    "adjudicated_utc": _utc(),
                    "deviation": "Operational postflight CPU activity is retained as an observation; numerical fits were complete and were not retried or changed."}
    adjudication["adjudication_elapsed_seconds"] = time.perf_counter() - started
    adjudication["elapsed_seconds"] = original["elapsed_seconds"] + adjudication["adjudication_elapsed_seconds"]
    if adjudication["elapsed_seconds"] > cap:
        raise TimeoutError("Adjudication finalization exceeds unchanged pilot compute cap")
    if sha256(original_path) != original_hash or any(_inventory(directory / unit_id) != hashes for unit_id, hashes in unit_hashes.items()):
        raise ValueError("Original receipt or unit artifacts changed during adjudication")
    write_json(directory / "completion_adjudication.json", adjudication)
    adjudication["adjudication_elapsed_seconds"] = time.perf_counter() - started
    adjudication["elapsed_seconds"] = original["elapsed_seconds"] + adjudication["adjudication_elapsed_seconds"]
    if adjudication["elapsed_seconds"] > cap:
        adjudication["status"] = "incomplete"
    write_json(directory / "completion_adjudication.json", adjudication)
    verify_stage(output_dir, "pilot", config, source_hash)
    # Charge the terminal receipt writes and structural verification, too. A
    # final write crossing the deadline must never leave an accepted completion.
    adjudication["adjudication_elapsed_seconds"] = time.perf_counter() - started
    adjudication["elapsed_seconds"] = original["elapsed_seconds"] + adjudication["adjudication_elapsed_seconds"]
    if adjudication["elapsed_seconds"] > cap:
        adjudication["status"] = "incomplete"
    write_json(directory / "completion_adjudication.json", adjudication)
    terminal_elapsed = original["elapsed_seconds"] + time.perf_counter() - started
    if terminal_elapsed > cap:
        adjudication.update(status="incomplete", elapsed_seconds=terminal_elapsed,
                            adjudication_elapsed_seconds=terminal_elapsed - original["elapsed_seconds"])
        write_json(directory / "completion_adjudication.json", adjudication)
        raise TimeoutError("Terminal adjudication finalization exceeds unchanged pilot compute cap")
    return adjudication


def lock_evaluation(output_dir: Path) -> dict[str, Any]:
    """Bind completed measured precursors before accessing the reserved pool."""
    lock_path = output_dir / "evaluation_lock.json"
    if lock_path.exists() or (output_dir / "evaluation").exists():
        raise FileExistsError("Reserved access has already been locked or attempted")
    config, source_hash = verify_freeze(output_dir)
    precursors = {}
    projections = {}
    factor = config["compute"]["pilot_projection_safety_factor"]
    for stage in ("pilot", "development"):
        receipt = verify_stage(output_dir, stage, config, source_hash)
        projection = (receipt["elapsed_seconds"] * config["pools"]["evaluation"]["units"]
                      / config["pools"][stage]["units"] * factor)
        projections[stage] = projection
        if projection > config["pools"]["evaluation"]["cap_seconds"]:
            raise ValueError(f"{stage} measured projection {projection:.3f}s exceeds reserved cap")
        precursors[stage] = _inventory(output_dir / stage)
    lock = {"created_utc": _utc(), "source_manifest_sha256": source_hash,
            "source_amendment_sha256": source_amendment_hash(output_dir),
            "precursor_artifacts": precursors, "projected_seconds": projections,
            "safety_factor": factor, "evaluation_cap_seconds": config["pools"]["evaluation"]["cap_seconds"],
            "selection": "No hyperparameter selection; all prospective settings retained"}
    write_json(lock_path, lock)
    return lock


def verify_lock(output_dir: Path, config: dict[str, Any], source_hash: str) -> str:
    """Verify the reserved lock and precursor bytes before generating data."""
    path = output_dir / "evaluation_lock.json"
    lock = read_json(path)
    if lock["source_manifest_sha256"] != source_hash:
        raise ValueError("Reserved lock belongs to a different source freeze")
    if lock.get("source_amendment_sha256") != source_amendment_hash(output_dir):
        raise ValueError("Reserved lock does not bind the current operational source amendment")
    for stage in ("pilot", "development"):
        receipt = verify_stage(output_dir, stage, config, source_hash)
        if _inventory(output_dir / stage) != lock["precursor_artifacts"][stage]:
            raise ValueError(f"Locked {stage} artifacts changed")
        projection = (receipt["elapsed_seconds"] * config["pools"]["evaluation"]["units"]
                      / config["pools"][stage]["units"]
                      * config["compute"]["pilot_projection_safety_factor"])
        if projection != lock["projected_seconds"][stage] or projection > config["pools"]["evaluation"]["cap_seconds"]:
            raise ValueError(f"Locked {stage} feasibility proof is invalid")
    return sha256(path)


def _load_apis() -> dict[str, Any]:
    from src.interpretation_n02_cases import fit_peak, fit_proxy, make_peak_unit, make_proxy_unit
    from src.interpretation_n02_diagnostics import peak_diagnostics, proxy_diagnostics

    return {"fit_peak": fit_peak, "fit_proxy": fit_proxy, "make_peak_unit": make_peak_unit,
            "make_proxy_unit": make_proxy_unit, "peak_diagnostics": peak_diagnostics,
            "proxy_diagnostics": proxy_diagnostics}


def _cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    return value


def _deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise TimeoutError("Prospective stage compute cap reached; no retries permitted")


def run_stage(output_dir: Path, stage: str) -> dict[str, Any]:
    """Generate, fit, diagnose and save the complete fixed pool within its cap."""
    if stage not in STAGES:
        raise ValueError(f"Unknown scientific stage: {stage}")
    started = time.perf_counter()
    config, source_hash = verify_freeze(output_dir)
    stage_dir = output_dir / stage
    if stage_dir.exists():
        raise FileExistsError(f"Stage already attempted; no overwrite or retry: {stage_dir}")
    lock_hash = verify_lock(output_dir, config, source_hash) if stage == "evaluation" else None
    if stage == "development":
        verify_stage(output_dir, "pilot", config, source_hash)
    cap = config["pools"][stage]["cap_seconds"]
    deadline = started + cap
    torch.set_num_threads(config["compute"]["cpu_threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observed_environment = environment(device)
    if observed_environment["competing_gpu_processes"]:
        raise RuntimeError("Competing GPU work detected; postpone timed stage: "
                           + repr(observed_environment["competing_gpu_processes"]))
    apis = _load_apis()
    stage_dir.mkdir(parents=True)
    completed: list[str] = []
    current_unit = None
    unit = None
    states: list[dict[str, Any]] = []
    receipt = {"stage": stage, "started_utc": _utc(), "status": "running",
               "source_manifest_sha256": source_hash, "evaluation_lock_sha256": lock_hash,
               "source_amendment_sha256": source_amendment_hash(output_dir),
               "cap_seconds": cap, "completed_units": completed,
               "environment": observed_environment,
               "timing_scope": "Stage entry through generation, fitting, diagnostics, artifact hashing and first receipt save"}
    write_json(stage_dir / "receipt.json", receipt)
    try:
        for case in config["cases"]:
            for index in range(config["pools"][stage]["units"]):
                _deadline(deadline)
                current_unit = f"{case}_{index:03d}"
                unit_dir = stage_dir / current_unit
                unit_dir.mkdir()
                unit_started = time.perf_counter()
                unit = None
                states = []
                if case == "proxy":
                    unit = apis["make_proxy_unit"](config, stage, index, device)
                    _deadline(deadline)
                    states.append(apis["fit_proxy"](unit, config, deadline=deadline))
                    _deadline(deadline)
                    metrics, details, arrays = apis["proxy_diagnostics"](states[0], unit, config)
                else:
                    unit = apis["make_peak_unit"](config, stage, index, case, device)
                    for seed in config["initializations"]:
                        _deadline(deadline)
                        states.append(apis["fit_peak"](unit, config, seed, deadline))
                    _deadline(deadline)
                    metrics, details, arrays = apis["peak_diagnostics"](states, unit, config)
                _deadline(deadline)
                if not isinstance(metrics, dict) or not all(isinstance(value, (int, float, np.number)) or value is None for value in metrics.values()):
                    raise TypeError("Diagnostics must return flat scalar metrics")
                torch.save({"unit": _cpu(unit), "states": _cpu(states)}, unit_dir / "payload.pt")
                write_json(unit_dir / "metrics.json", metrics)
                write_json(unit_dir / "details.json", details)
                np.savez_compressed(unit_dir / "arrays.npz", **arrays)
                _deadline(deadline)
                write_json(unit_dir / "receipt.json", {
                    "status": "complete", "case": case, "index": index, "fits": len(states),
                    "elapsed_seconds": time.perf_counter() - unit_started,
                    "undefined_metrics": [key for key, value in _json_safe(metrics).items() if value is None],
                })
                _deadline(deadline)
                completed.append(current_unit)
                print(f"{stage}: {current_unit} complete ({time.perf_counter() - started:.2f}s)", flush=True)
        receipt["environment_end"] = gpu_end_snapshot(device, observed_environment)
        write_json(stage_dir / "stage_summary.json", {
            "stage": stage, "units_per_case": config["pools"][stage]["units"],
            "completed_units": completed, "initializations": config["initializations"],
            "proxy_initializations": "Unique ridge solution; one deterministic fit per trajectory",
            "source_manifest_sha256": source_hash,
            "source_amendment_sha256": source_amendment_hash(output_dir),
        })
        _deadline(deadline)
        write_json(stage_dir / "artifact_manifest.json", {
            "artifacts": _inventory(stage_dir, {"receipt.json", "artifact_manifest.json"})})
        _deadline(deadline)
        receipt.update(status="complete", elapsed_seconds=time.perf_counter() - started,
                       artifact_manifest_sha256=sha256(stage_dir / "artifact_manifest.json"))
        write_json(stage_dir / "receipt.json", receipt)
        _deadline(deadline)
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["finished_utc"] = _utc()
        write_json(stage_dir / "receipt.json", receipt)
        _deadline(deadline)
    except BaseException as error:
        receipt.update(status="incomplete" if isinstance(error, (TimeoutError, KeyboardInterrupt)) else "error",
                       elapsed_seconds=time.perf_counter() - started, failed_unit=current_unit,
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                       finished_utc=_utc())
        if current_unit and current_unit not in completed:
            if unit is not None:
                torch.save({"unit": _cpu(unit), "states": _cpu(states)},
                           stage_dir / current_unit / "incomplete_payload.pt")
            write_json(stage_dir / current_unit / "receipt.json", {
                "status": receipt["status"], "fits_retained": len(states),
                "error_type": type(error).__name__, "error": str(error)})
        write_json(stage_dir / "receipt.json", receipt)
        raise
    return receipt


def aggregate_units(records: list[dict[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    """Bootstrap independent units, retaining every missing/nonfinite outcome.

    Diagnostics already average seeds and initialization pairs within units.
    All metrics use the same resampled unit indices to preserve pairing.
    """
    if not records:
        raise ValueError("No independent units to aggregate")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(records), size=(draws, len(records)))
    result = {}
    for key in sorted(set().union(*(record.keys() for record in records))):
        values = np.asarray([record.get(key) if record.get(key) is not None else np.nan
                             for record in records], dtype=float)
        valid = np.isfinite(values)
        complete = bool(valid.all())
        item = {"n_units": len(records), "n_finite": int(valid.sum()),
                "complete": complete, "mean": None, "ci95": None, "unit_sd": None}
        if complete:
            samples = values[indices].mean(axis=1)
            item.update(mean=float(values.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                        unit_sd=float(values.std(ddof=1)) if len(values) > 1 else 0.0)
        result[key] = item
    return result


def analyze(output_dir: Path, stage: str = "evaluation", summary_dir: Path | None = None) -> dict[str, Any]:
    """Write compact unit tables and paired-unit bootstrap summaries."""
    config, source_hash = verify_freeze(output_dir)
    receipt = verify_stage(output_dir, stage, config, source_hash)
    if stage == "evaluation":
        lock_hash = verify_lock(output_dir, config, source_hash)
        if receipt["evaluation_lock_sha256"] != lock_hash:
            raise ValueError("Evaluation receipt does not bind the current reserved lock")
    destination = summary_dir or output_dir / f"analysis_{stage}"
    if destination.exists():
        raise FileExistsError(f"Analysis destination already exists: {destination}")
    records = []
    aggregates = {}
    for case in config["cases"]:
        metrics = []
        for index in range(config["pools"][stage]["units"]):
            data = read_json(output_dir / stage / f"{case}_{index:03d}" / "metrics.json")
            metrics.append(data)
            records.append({"case": case, "unit": index, **data})
        aggregates[case] = aggregate_units(metrics, config["diagnostics"]["bootstrap_draws"],
                                           config["diagnostics"]["bootstrap_seed"])
    destination.mkdir(parents=True)
    summary = {"stage": stage, "source_manifest_sha256": source_hash,
               "source_amendment_sha256": source_amendment_hash(output_dir),
               "stage_receipt_sha256": sha256(output_dir / stage / "receipt.json"),
               "completion_adjudication_sha256": (sha256(output_dir / stage / "completion_adjudication.json")
                                                    if (output_dir / stage / "completion_adjudication.json").is_file() else None),
               "stage_artifact_manifest_sha256": sha256(output_dir / stage / "artifact_manifest.json"),
               "bootstrap_draws": config["diagnostics"]["bootstrap_draws"],
               "bootstrap_seed": config["diagnostics"]["bootstrap_seed"],
               "uncertainty_unit": "Independent function/trajectory; initialization metrics averaged within unit",
               "missing_policy": "Any missing/nonfinite unit suppresses that metric's aggregate; no survivor-only means",
               "cases": aggregates}
    write_json(destination / "summary.json", summary)
    fields = ["case", "unit"] + sorted(set().union(*(row.keys() for row in records)) - {"case", "unit"})
    with (destination / "units.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint; exceptions produce nonzero status without retrying."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("freeze", *STAGES, "lock", "analyze", "adjudicate-postflight"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--analyze-stage", choices=STAGES, default="evaluation")
    parser.add_argument("--summary-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.stage == "freeze":
            result = freeze_run(args.output_dir, args.config, args.protocol)
        elif args.stage == "lock":
            result = lock_evaluation(args.output_dir)
        elif args.stage == "analyze":
            result = analyze(args.output_dir, args.analyze_stage, args.summary_dir)
        elif args.stage == "adjudicate-postflight":
            result = adjudicate_completed_postflight(args.output_dir)
        else:
            result = run_stage(args.output_dir, args.stage)
        print(json.dumps(_json_safe({"stage": args.stage, "status": result.get("status", "complete"),
                                    "output_dir": str(args.output_dir.resolve())}), sort_keys=True), flush=True)
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
