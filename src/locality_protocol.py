"""Prospective locality-v2 matrices, provenance and procedural holdout guards."""

from __future__ import annotations

import itertools
import math
import re
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.methodology_protocol import environment, object_hash, write_json
from src.methodology_protocol import file_hash as _raw_file_hash, read_json as _raw_read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/vm_locality_v2.json"
SOURCE_PATHS = (
    "configs/vm_locality_v2.json", "docs/vm_locality_v2_protocol.md",
    "src/locality_models.py", "src/locality_tasks.py", "src/locality_training.py",
    "src/locality_protocol.py", "scripts/run_vm_locality.py",
    "scripts/analyze_vm_locality.py", "src/circular_layers_v2.py",
    "src/methodology_tasks.py", "src/methodology_protocol.py",
)

_EVIDENCE_CACHE: ContextVar[dict[tuple[str, str], tuple[int, int, Any]] | None] = ContextVar("locality_evidence_cache", default=None)


def _memoized_evidence(function):
    """Share exact-byte reads only during one synchronous verification call."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _EVIDENCE_CACHE.get() is not None:
            return function(*args, **kwargs)
        token = _EVIDENCE_CACHE.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _EVIDENCE_CACHE.reset(token)
    return wrapped


def _cached_read(path: Path, kind: str, reader):
    """Invalidate within-call memoized reads if size or modification time changes."""
    cache = _EVIDENCE_CACHE.get()
    if cache is None:
        return reader(path)
    resolved = path.resolve()
    state = resolved.stat()
    key = (kind, str(resolved))
    signature = (state.st_mtime_ns, state.st_size)
    if key not in cache or cache[key][:2] != signature:
        value = reader(resolved)
        after = resolved.stat()
        if (after.st_mtime_ns, after.st_size) != signature:
            raise ValueError("Evidence changed while being read")
        cache[key] = (*signature, value)
    return cache[key][2]


def file_hash(path: Path) -> str:
    """Hash bytes, memoized only inside a current top-level verification call."""
    return _cached_read(Path(path), "sha256", _raw_file_hash)


def read_json(path: Path) -> Any:
    """Read strict JSON, sharing read-only objects within one verification call."""
    return _cached_read(Path(path), "json", _raw_read_json)


def source_manifest(root: Path = ROOT) -> dict[str, str]:
    """Hash every scientific and orchestration dependency, including frozen imports."""
    return {name: file_hash(root / name) for name in SOURCE_PATHS}


def snapshot_sources(output: Path, root: Path = ROOT) -> dict[str, str]:
    """Preserve exact source bytes without overwriting an existing snapshot."""
    hashes = source_manifest(root)
    for name in hashes:
        target = output / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((root / name).read_bytes())
    return hashes


def planned_runs(config: Mapping[str, Any], phase: str,
                 learning_rates: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """Enumerate all cells before fitting; pilot timing rows never select rates."""
    if phase not in ("pilot", "development", "confirmation"):
        raise ValueError("Unknown locality phase")
    if phase != "pilot":
        validate_learning_rates(learning_rates, config)
    task = config["tasks"]
    panels = [("main", task["label_budgets"])] if phase != "pilot" else [
        ("selection", task["pilot_label_budgets"]),
        ("timing", [task["pilot_timing_label_budget"]]),
    ]
    result: list[dict[str, Any]] = []
    for role, budgets in panels:
        for family, replicate, noise, budget, init, model in itertools.product(
            task["families"], range(task["replicates"][phase]), task["noise_levels"],
            budgets, range(task["initializations"][phase]), config["models"],
        ):
            if model["kind"] == "ridge":
                rates = [None]
            elif phase == "pilot":
                rates = (config["training"]["pilot_learning_rates"] if role == "selection"
                         else [task["pilot_timing_learning_rate"]])
            else:
                if learning_rates is None or model["id"] not in learning_rates:
                    raise ValueError("All adaptive rates must be selected before a larger phase")
                rates = [learning_rates[model["id"]]]
            for rate in rates:
                cell = {"phase": phase, "role": role, "family": family,
                        "replicate": replicate, "noise_std": noise, "total_labels": budget,
                        "initialization": init, "model_id": model["id"], "kind": model["kind"],
                        "learning_rate": rate}
                cell["id"] = object_hash(cell)[:24]
                result.append(cell)
    # Order is independent of scores and model RNG, to distribute device drift.
    rng = np.random.default_rng(np.random.SeedSequence([
        config["seeds"]["analysis"], 73, ("pilot", "development", "confirmation").index(phase)]))
    return [result[int(i)] for i in rng.permutation(len(result))]


def checked_path(name: str, root: Path = ROOT) -> Path:
    """Resolve an evidence path while rejecting traversal outside the repository."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path escapes repository")
    return path


def verify_dependencies(hashes: Mapping[str, str], root: Path = ROOT) -> None:
    """Require exact bytes for every bound artifact."""
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("Missing evidence dependencies")
    for name, digest in hashes.items():
        if (not isinstance(name, str) or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("Invalid evidence path or SHA256")
        path = checked_path(name, root)
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f"Evidence changed or absent: {name}")


@_memoized_evidence
def verify_bound_record(path: Path, config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Check a completed record's source, configuration and downstream dependencies."""
    record = read_json(checked_path(str(path), root))
    if not isinstance(record, dict):
        raise ValueError("A bound record must be a JSON object")
    if record.get("config_hash") != object_hash(config):
        raise ValueError("Configuration binding mismatch")
    if record.get("source_hashes") != source_manifest(root):
        raise ValueError("Source binding mismatch")
    verify_dependencies(record.get("evidence_hashes", {}), root)
    return record


def validate_learning_rates(rates: Mapping[str, float] | None,
                            config: Mapping[str, Any]) -> None:
    """Require exactly the four adaptive models and prospectively piloted rates."""
    models = {model["id"] for model in config["models"] if model["kind"] != "ridge"}
    if not isinstance(rates, Mapping) or set(rates) != models:
        raise ValueError("Learning-rate map must contain every adaptive model exactly once")
    for rate in rates.values():
        if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                or not math.isfinite(rate) or rate <= 0
                or rate not in config["training"]["pilot_learning_rates"]):
            raise ValueError("Learning rates must be finite positive pilot candidates")


def validate_choices(choices: Mapping[str, str], config: Mapping[str, Any]) -> None:
    """Require a declared VP strategy and a fixed comparator of equal parameter count."""
    if not isinstance(choices, Mapping) or set(choices) != {"vp_model", "capacity_comparator"}:
        raise ValueError("Choices require vp_model and capacity_comparator exactly")
    if any(not isinstance(value, str) for value in choices.values()):
        raise ValueError("Model choices must be strings")
    candidates = {model["id"]: model for model in config["models"]}
    vp = candidates.get(choices["vp_model"])
    comparator = candidates.get(choices["capacity_comparator"])
    if vp is None or vp["kind"] != "vp" or comparator is None or comparator["kind"] != "ridge":
        raise ValueError("Choices must name registered VP and fixed models")
    if vp["parameters"] != comparator["parameters"]:
        raise ValueError("Selected capacity comparator must match VP fitted parameter count")


def _finite_nonnegative(value: Any, name: str) -> float:
    """Reject booleans and nonnumeric/nonfinite or negative evidence fields."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _relative(path: Path, root: Path) -> str:
    return checked_path(str(path), root).relative_to(root.resolve()).as_posix()


def _bound_path(record: Mapping[str, Any], name: str, root: Path) -> Path:
    """Resolve an explicit canonical path that is itself in the evidence closure."""
    if not isinstance(name, str):
        raise ValueError("Evidence references must be path strings")
    path = checked_path(name, root)
    if name != _relative(path, root):
        raise ValueError("Evidence references must use repository-relative canonical paths")
    if record.get("evidence_hashes", {}).get(name) != file_hash(path):
        raise ValueError(f"Referenced artifact is not hash-bound: {name}")
    return path


def _typed_record(path: Path, kind: str, config: Mapping[str, Any], root: Path,
                  *, passed: bool = False) -> dict[str, Any]:
    record = verify_bound_record(path, config, root)
    if record.get("kind") != kind or record.get("complete") is not True:
        raise ValueError(f"Expected a complete {kind} record")
    if passed and record.get("all_passed") is not True:
        raise ValueError(f"Failed {kind} prerequisite")
    return record


def _closure_contains(record: Mapping[str, Any], dependency: Mapping[str, Any]) -> None:
    if any(record.get("evidence_hashes", {}).get(name) != digest
           for name, digest in dependency["evidence_hashes"].items()):
        raise ValueError("Incomplete or inconsistent prerequisite evidence closure")


@_memoized_evidence
def verify_run_manifest(path: Path, config: Mapping[str, Any], root: Path = ROOT,
                        phase: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify a complete exact matrix and its immutable, explicitly bound fit records.

    This reads evidence only. Artifact tensor contents require the separate
    independent numerical analysis; they are never loaded or executed here.
    """
    manifest = _typed_record(path, "locality_run_v2", config, root)
    actual_phase = manifest.get("phase")
    if phase is not None and actual_phase != phase:
        raise ValueError(f"Expected a {phase} invocation")
    if manifest.get("status") != "completed" or manifest.get("deadline_respected") is not True:
        raise ValueError("A complete within-deadline invocation is required")
    expected = planned_runs(config, actual_phase, manifest.get("learning_rates"))
    rows = manifest.get("rows", [])
    if (not isinstance(rows, list) or [row.get("cell") for row in rows] != expected
            or manifest.get("planned_count") != len(expected)
            or manifest.get("completed_count") != len(expected)):
        raise ValueError("Invocation does not contain the exact complete canonical matrix")
    if actual_phase == "pilot" and manifest.get("learning_rates") is not None:
        raise ValueError("Pilot runs cannot use preselected learning rates")
    ceiling = config["phases"][f"{actual_phase}_ceiling_seconds"]
    started = _finite_nonnegative(manifest.get("started_unix"), "started_unix")
    ended = _finite_nonnegative(manifest.get("ended_unix"), "ended_unix")
    deadline = _finite_nonnegative(manifest.get("deadline_unix"), "deadline_unix")
    elapsed = _finite_nonnegative(manifest.get("elapsed_seconds"), "elapsed_seconds")
    if (manifest.get("phase_ceiling_seconds") != ceiling
            or not math.isclose(deadline, started + ceiling, rel_tol=0, abs_tol=1e-6)
            or ended < started or ended > deadline
            or not math.isclose(elapsed, ended - started, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Invocation timing disagrees with its immutable phase deadline")
    receipt_path = path.resolve().parent / "completion.json"
    if not receipt_path.is_file():
        raise ValueError("Missing post-manifest completion receipt")
    receipt = read_json(receipt_path)
    completion = _finite_nonnegative(receipt.get("ended_unix"), "completion ended_unix")
    completion_elapsed = _finite_nonnegative(receipt.get("elapsed_seconds"), "completion elapsed_seconds")
    if (receipt.get("manifest_hash") != file_hash(path) or receipt.get("deadline_respected") is not True
            or completion < ended or completion > deadline
            or not math.isclose(completion_elapsed, completion - started, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Final manifest write did not finish within the bound phase deadline")
    # Receipt is produced after the final immutable manifest. Enrich only this
    # returned view so downstream records bind it without a circular file hash.
    manifest = {**manifest, "evidence_hashes": {**manifest["evidence_hashes"],
                                               _relative(receipt_path, root): file_hash(receipt_path)}}
    summaries = []
    for row in rows:
        if row.get("status") != "completed":
            raise ValueError("Noncomplete fit cannot be omitted from a successful matrix")
        summary_path = _bound_path(manifest, row.get("summary_path"), root)
        artifact_path = _bound_path(manifest, row.get("artifact_path"), root)
        _bound_path(manifest, row.get("task_path"), root)
        if row.get("summary_hash") != file_hash(summary_path) or row.get("artifact_hash") != file_hash(artifact_path):
            raise ValueError("Fit row hashes disagree with bound artifacts")
        directory = path.resolve().parent / "fits" / row["cell"]["id"]
        if summary_path != directory / "summary.json" or artifact_path != directory / "artifacts.pt":
            raise ValueError("Each canonical fit must retain its own immutable artifact directory")
        summary = read_json(summary_path)
        if summary.get("status") != "completed" or summary.get("model_id") != row["cell"]["model_id"]:
            raise ValueError("Fit summary status/model identity mismatch")
        summaries.append(summary)
    return manifest, summaries


@_memoized_evidence
def verify_pilot_selection(path: Path, config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Replay complete pilot rate selection from the bound noisy validation summaries."""
    record = _typed_record(path, "locality_pilot_selection_v2", config, root)
    validate_learning_rates(record.get("learning_rates"), config)
    manifest, summaries = verify_run_manifest(_bound_path(record, record.get("input_manifest"), root), config, root, "pilot")
    _closure_contains(record, manifest)
    scores = record.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(record["learning_rates"]):
        raise ValueError("Missing pilot candidate scores")
    expected_count = (len(config["tasks"]["families"]) * len(config["tasks"]["noise_levels"])
                      * config["tasks"]["replicates"]["pilot"] * config["tasks"]["initializations"]["pilot"]
                      * len(config["tasks"]["pilot_label_budgets"]))
    for model_id in record["learning_rates"]:
        candidates = []
        for rate in config["training"]["pilot_learning_rates"]:
            values = [_finite_nonnegative(summary.get("best_validation_mse"), "pilot validation MSE")
                      for row, summary in zip(manifest["rows"], summaries)
                      if row["cell"]["role"] == "selection" and row["cell"]["model_id"] == model_id
                      and row["cell"]["learning_rate"] == rate]
            if len(values) != expected_count:
                raise ValueError("Incomplete pilot candidate observations")
            candidates.append({"learning_rate": rate, "mean_validation_mse": float(np.mean(values)), "count": len(values)})
        if scores[model_id] != candidates:
            raise ValueError("Pilot scores disagree with complete noisy-validation evidence")
        winner = min(candidates, key=lambda item: (item["mean_validation_mse"], item["learning_rate"]))["learning_rate"]
        if record["learning_rates"][model_id] != winner:
            raise ValueError("Learning rate does not follow the prescribed pilot selection rule")
    return record


@_memoized_evidence
def verify_feasibility(path: Path, config: Mapping[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Check both phase projections against measured evidence from the same pilot."""
    record = _typed_record(path, "locality_feasibility_v2", config, root, passed=True)
    selection = verify_pilot_selection(_bound_path(record, record.get("pilot_selection"), root), config, root)
    if record.get("input_manifest") != selection["input_manifest"]:
        raise ValueError("Feasibility and rate selection must bind the same pilot")
    manifest, summaries = verify_run_manifest(_bound_path(record, record.get("input_manifest"), root), config, root, "pilot")
    _closure_contains(record, manifest)
    _closure_contains(record, selection)
    estimates, measured = record.get("estimates"), record.get("measured")
    if not isinstance(estimates, dict) or set(estimates) != {"development", "confirmation"} or not isinstance(measured, dict):
        raise ValueError("Both development and confirmation feasibility estimates are required")
    fixed = [_finite_nonnegative(row.get("fit_call_seconds"), "fixed fit seconds") for row in manifest["rows"]
             if row["cell"]["role"] == "timing" and row["cell"]["kind"] == "ridge"]
    overhead = [max(0., _finite_nonnegative(row.get("wall_seconds"), "cell wall seconds")
                    - _finite_nonnegative(summary.get("fit_seconds"), "fit seconds"))
                for row, summary in zip(manifest["rows"], summaries)]
    overshoot = max(_finite_nonnegative(summary.get("overshoot_seconds"), "overshoot seconds")
                    for row, summary in zip(manifest["rows"], summaries) if row["cell"]["kind"] != "ridge")
    observed = {"fixed_maxshape_call_p90": float(np.quantile(fixed, .9)),
                "nonfit_overhead_p90": float(np.quantile(overhead, .9)), "max_overshoot": overshoot}
    if set(measured) != set(observed) or any(not math.isclose(_finite_nonnegative(measured[name], name), value, rel_tol=1e-12, abs_tol=1e-12)
                                             for name, value in observed.items()):
        raise ValueError("Feasibility measurements disagree with bound pilot timing evidence")
    for phase, estimate in estimates.items():
        rows = planned_runs(config, phase, selection["learning_rates"])
        adaptive = sum(row["kind"] != "ridge" for row in rows)
        seconds = adaptive * (config["training"]["primary_time_seconds"] + overshoot)
        seconds += (len(rows) - adaptive) * observed["fixed_maxshape_call_p90"] + len(rows) * observed["nonfit_overhead_p90"] + 60
        ceiling = config["phases"][f"{phase}_ceiling_seconds"]
        if (estimate.get("records") != len(rows) or estimate.get("adaptive") != adaptive
                or estimate.get("fixed") != len(rows) - adaptive or estimate.get("ceiling_seconds") != ceiling
                or estimate.get("passed") is not True or seconds > ceiling
                or not math.isclose(_finite_nonnegative(estimate.get("projected_seconds"), "projected seconds"), seconds, rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError(f"Invalid or failed {phase} feasibility projection")
    return record


def _verify_development(path: Path, config: Mapping[str, Any], rates: Mapping[str, float],
                        root: Path) -> dict[str, Any]:
    record = _typed_record(path, "locality_analysis_v2", config, root)
    if record.get("phase") != "development":
        raise ValueError("Development analysis required")
    if not isinstance(record.get("audit"), dict) or record["audit"].get("all_passed") is not True:
        raise ValueError("Development requires a passing artifact audit, independent of research gates")
    if record.get("learning_rates") != rates:
        raise ValueError("Development analysis rates disagree with the pilot selection")
    validate_choices(record.get("choices"), config)
    manifest, summaries = verify_run_manifest(_bound_path(record, record.get("input_manifest"), root), config, root, "development")
    _closure_contains(record, manifest)
    if manifest.get("learning_rates") != rates:
        raise ValueError("Development invocation rates disagree with the pilot selection")
    scores = {}
    for model in config["models"]:
        family_scores = []
        for family in config["tasks"]["periodic_families"]:
            task_scores = []
            for replicate in range(config["tasks"]["replicates"]["development"]):
                values = [_finite_nonnegative(summary.get("primary_error"), "development primary error")
                          for row, summary in zip(manifest["rows"], summaries)
                          if row["cell"]["model_id"] == model["id"] and row["cell"]["family"] == family
                          and row["cell"]["replicate"] == replicate]
                task_scores.append(float(np.mean(values)))
            family_scores.append(float(np.mean(task_scores)))
        scores[model["id"]] = float(np.mean(family_scores))
    vp = min((model for model in config["models"] if model["kind"] == "vp"), key=lambda model: scores[model["id"]])
    comparator = min((model for model in config["models"] if model["kind"] == "ridge"
                      and model["parameters"] == vp["parameters"]), key=lambda model: scores[model["id"]])
    if record["choices"] != {"vp_model": vp["id"], "capacity_comparator": comparator["id"]}:
        raise ValueError("Development choices do not follow the complete balanced selection rule")
    return record


def _verify_validation(path: Path, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    record = _typed_record(path, "locality_validation_v2", config, root, passed=True)
    tests = {file.relative_to(root).as_posix(): file_hash(file) for file in sorted((root / "tests").rglob("*.py"))}
    if not tests or record.get("test_source_hashes") != tests:
        raise ValueError("Validation must bind the current complete Python test sources")
    if any(record["evidence_hashes"].get(name) != digest for name, digest in tests.items()):
        raise ValueError("Validation test sources are missing from evidence hashes")
    suite = record.get("full_suite")
    if (not isinstance(suite, dict) or suite.get("command") != ["py", "-3.11", "-m", "pytest", "tests/", "-q"]
            or any(type(suite.get(key)) is not int or suite[key] != 0 for key in ("exit_code", "failed", "errors"))
            or isinstance(suite.get("passed"), bool) or not isinstance(suite.get("passed"), int)
            or suite["passed"] <= 0):
        raise ValueError("Successful full-suite validation proof is required")
    log = _bound_path(record, suite.get("log_path"), root).read_text(encoding="utf-8-sig")
    passed_counts = re.findall(r"\b(\d+) passed\b", log)
    failure_counts = re.findall(r"\b([1-9]\d*) (?:failed|errors?)\b", log)
    if not passed_counts or int(passed_counts[-1]) != suite["passed"] or failure_counts:
        raise ValueError("Full-suite log does not support its declared passing result")
    return record


def _verify_independent_audit(path: Path, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    record = _typed_record(path, "locality_independent_audit_v2", config, root, passed=True)
    if record.get("checks") != {"numerical": True, "protocol": True}:
        raise ValueError("Independent numerical and protocol checks are both required")
    reports = record.get("reports")
    if not isinstance(reports, dict) or set(reports) != {"numerical", "protocol"}:
        raise ValueError("Independent audit requires both detailed reports")
    if len(set(reports.values())) != 2:
        raise ValueError("Numerical and protocol reviews must be separate reports")
    for filename in reports.values():
        review = read_json(_bound_path(record, filename, root))
        if (not isinstance(review, dict) or review.get("all_passed") is not True
                or review.get("source_hashes") != source_manifest(root)
                or not isinstance(review.get("scope"), str) or not review["scope"].strip()
                or not isinstance(review.get("findings"), list)):
            raise ValueError("Review report must bind current source hashes and passing findings")
    return record


def _prerequisites(inputs: Mapping[str, str | Path], config: Mapping[str, Any],
                   root: Path) -> dict[str, dict[str, Any]]:
    required = {"pilot_selection", "development_analysis", "feasibility", "independent_audit", "validation"}
    if not isinstance(inputs, Mapping) or set(inputs) != required:
        raise ValueError("All five confirmation prerequisites are required")
    paths = {name: checked_path(str(filename), root) for name, filename in inputs.items()}
    if len(set(paths.values())) != len(required):
        raise ValueError("Confirmation prerequisites must be distinct records")
    pilot = verify_pilot_selection(paths["pilot_selection"], config, root)
    feasibility = verify_feasibility(paths["feasibility"], config, root)
    if feasibility["pilot_selection"] != _relative(paths["pilot_selection"], root):
        raise ValueError("Feasibility must bind this exact pilot selection record")
    if feasibility["input_manifest"] != pilot["input_manifest"]:
        raise ValueError("Prerequisites refer to different pilot invocations")
    development = _verify_development(paths["development_analysis"], config, pilot["learning_rates"], root)
    pilot_run = read_json(checked_path(pilot["input_manifest"], root))
    development_run = read_json(checked_path(development["input_manifest"], root))
    expected_inputs = {_relative(paths[name], root): file_hash(paths[name]) for name in ("pilot_selection", "feasibility")}
    if development_run.get("prerequisite_hashes") != expected_inputs:
        raise ValueError("Development invocation must bind this exact pilot selection and feasibility")
    if pilot_run["ended_unix"] > development_run["started_unix"]:
        raise ValueError("Pilot must finish before development begins")
    return {"pilot_selection": pilot, "feasibility": feasibility,
            "development_analysis": development,
            "independent_audit": _verify_independent_audit(paths["independent_audit"], config, root),
            "validation": _verify_validation(paths["validation"], config, root)}


@_memoized_evidence
def verify_confirmation_lock(path: str | Path | None, config: Mapping[str, Any],
                             root: str | Path | None = None) -> dict[str, Any]:
    """Reject generation/restoration of reserved targets without complete locked evidence.

    This is a procedural access guard, not a secrecy or authentication mechanism.
    """
    if path is None:
        raise ValueError("Confirmation requires a verified locality-v2 lock")
    repo = Path(root) if root is not None else ROOT
    lock_path = checked_path(str(path), repo)
    sidecar = lock_path.with_suffix(lock_path.suffix + ".sha256")
    if not lock_path.is_file() or not sidecar.is_file() or sidecar.read_text().strip() != file_hash(lock_path):
        raise ValueError("Missing or changed lock checksum")
    lock = verify_bound_record(lock_path, config, repo)
    if (lock.get("kind") != "locality_confirmation_lock_v2" or lock.get("complete") is not True
            or lock.get("protocol_id") != config["protocol_id"]):
        raise ValueError("Not a complete locality confirmation lock")
    if not isinstance(lock.get("inputs"), dict):
        raise ValueError("Missing confirmation prerequisite paths")
    for filename in lock["inputs"].values():
        _bound_path(lock, filename, repo)
    records = _prerequisites(lock["inputs"], config, repo)
    for record in records.values():
        _closure_contains(lock, record)
    validate_learning_rates(lock.get("learning_rates"), config)
    validate_choices(lock.get("choices"), config)
    if (lock["learning_rates"] != records["pilot_selection"]["learning_rates"]
            or lock["choices"] != records["development_analysis"]["choices"]):
        raise ValueError("Locked rates or choices disagree with their bound prerequisite records")
    if "lock_sha256" in lock:
        raise ValueError("Lock digest is computed on verification and must not be serialized")
    return {**lock, "lock_sha256": file_hash(lock_path)}


@_memoized_evidence
def create_confirmation_lock(output: Path, config: Mapping[str, Any], inputs: Mapping[str, Path],
                             root: Path = ROOT) -> dict[str, Any]:
    """Bind complete audited pilot/development evidence before any holdout access."""
    output = checked_path(str(output), root)
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError("Confirmation lock or checksum already exists")
    records = _prerequisites(inputs, config, root)
    evidence = {}
    for name, path in inputs.items():
        evidence[_relative(path, root)] = file_hash(path)
        evidence.update(records[name]["evidence_hashes"])
    lock = {"kind": "locality_confirmation_lock_v2", "complete": True,
            "protocol_id": config["protocol_id"], "config_hash": object_hash(config),
            "source_hashes": source_manifest(root), "environment": environment(),
            "inputs": {name: _relative(path, root)
                       for name, path in inputs.items()}, "evidence_hashes": evidence,
            "learning_rates": records["pilot_selection"]["learning_rates"],
            "choices": records["development_analysis"]["choices"]}
    write_json(output, lock)
    with output.with_suffix(output.suffix + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(file_hash(output) + "\n")
    return verify_confirmation_lock(output, config, root)
