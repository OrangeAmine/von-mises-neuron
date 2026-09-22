"""Protocol guards tested with inert fixture APIs, never scientific pool draws."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_vm_n02_interpretation as runner


@pytest.fixture
def frozen_run(tmp_path, monkeypatch):
    """Freeze disposable source/config bytes and install deterministic fake APIs."""
    config = {
        "version": "fixture",
        "pools": {stage: {"seed_offset": offset, "units": 2, "cap_seconds": 1000}
                  for stage, offset in zip(runner.STAGES, [100, 200, 300])},
        "cases": ["isolated", "redundant", "proxy"],
        "initializations": [11, 29, 47],
        "compute": {"dtype": "float64", "cpu_threads": 1, "pilot_projection_safety_factor": 1.5},
        "diagnostics": {"bootstrap_draws": 100, "bootstrap_seed": 2},
    }
    config_path = tmp_path / "config.json"
    runner.write_json(config_path, config)
    protocol_path = tmp_path / "protocol.md"
    protocol_path.write_text("Prospective fixture only.\n", encoding="utf-8")
    scientific_path = tmp_path / "scientific.py"
    scientific_path.write_text("# frozen fixture\n", encoding="utf-8")
    runner_path = tmp_path / "runner.py"
    runner_path.write_text("# original operational runner fixture\n", encoding="utf-8")
    monkeypatch.setattr(runner, "RUNNER_SOURCE", runner_path)
    out = tmp_path / "run"
    runner.freeze_run(out, config_path, protocol_path, (scientific_path, runner_path))
    calls = []

    def make_peak(config, stage, index, case, device):
        calls.append((stage, index, case))
        return {"index": index, "case": case, "x": torch.tensor([index], dtype=torch.float64, device=device)}

    def make_proxy(config, stage, index, device):
        return make_peak(config, stage, index, "proxy", device)

    def fit_peak(unit, config, seed, deadline):
        return {"seed": seed, "w": torch.tensor([1.0], dtype=torch.float64)}

    def fit_proxy(unit, config, deadline=None):
        return {"w": torch.tensor([1.0], dtype=torch.float64)}

    def diagnostic(states, unit, config):
        return ({"error": float(unit["index"]), "paired_effect": float(unit["index"] + 2)},
                {"fixture": True}, {"curve": np.arange(3.0)})

    apis = {"make_peak_unit": make_peak, "make_proxy_unit": make_proxy,
            "fit_peak": fit_peak, "fit_proxy": fit_proxy,
            "peak_diagnostics": diagnostic, "proxy_diagnostics": diagnostic}
    monkeypatch.setattr(runner, "_load_apis", lambda: apis)
    monkeypatch.setattr(runner, "environment", lambda device: {"competing_gpu_processes": [], "device": str(device)})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return {"output": out, "config": config, "config_path": config_path,
            "source": scientific_path, "runner_source": runner_path, "calls": calls, "apis": apis}


def test_freeze_binds_live_source_and_frozen_copies(frozen_run):
    out = frozen_run["output"]
    config, source_hash = runner.verify_freeze(out)
    assert config == frozen_run["config"]
    assert len(source_hash) == 64
    frozen_run["source"].write_text("# unexpected change\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        runner.run_stage(out, "pilot")
    assert not frozen_run["calls"]
    assert not (out / "pilot").exists()


def test_frozen_copy_tampering_blocks_generation(frozen_run):
    out = frozen_run["output"]
    (out / "frozen/protocol.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen copy changed"):
        runner.run_stage(out, "pilot")
    assert not frozen_run["calls"]


def test_complete_pipeline_preserves_matrix_and_payload_and_analysis(frozen_run):
    out = frozen_run["output"]
    for stage in ("pilot", "development"):
        receipt = runner.run_stage(out, stage)
        assert receipt["status"] == "complete"
        assert len(receipt["completed_units"]) == 6
    lock = runner.lock_evaluation(out)
    assert all(value < 1000 for value in lock["projected_seconds"].values())
    runner.run_stage(out, "evaluation")
    assert len(frozen_run["calls"]) == 18
    config, source_hash = runner.verify_freeze(out)
    runner.verify_stage(out, "evaluation", config, source_hash)
    payload = torch.load(out / "evaluation/isolated_000/payload.pt", weights_only=False)
    assert len(payload["states"]) == 3
    assert payload["unit"]["x"].device.type == "cpu"
    proxy = torch.load(out / "evaluation/proxy_000/payload.pt", weights_only=False)
    assert len(proxy["states"]) == 1
    summary = runner.analyze(out)
    item = summary["cases"]["proxy"]["paired_effect"]
    assert item["mean"] == 2.5 and item["n_units"] == 2 and item["complete"]
    assert (out / "analysis_evaluation/units.csv").exists()


def test_reserved_generation_requires_completed_precursors_and_lock(frozen_run):
    out = frozen_run["output"]
    with pytest.raises(FileNotFoundError):
        runner.run_stage(out, "evaluation")
    with pytest.raises(FileNotFoundError):
        runner.lock_evaluation(out)
    assert frozen_run["calls"] == []
    assert not (out / "evaluation").exists()


def test_all_stages_and_analysis_refuse_overwrite(frozen_run):
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    before = list(frozen_run["calls"])
    with pytest.raises(FileExistsError):
        runner.run_stage(out, "pilot")
    assert frozen_run["calls"] == before
    runner.analyze(out, "pilot")
    with pytest.raises(FileExistsError):
        runner.analyze(out, "pilot")
    with pytest.raises(FileExistsError):
        runner.freeze_run(out, frozen_run["config_path"], out / "frozen/protocol.md", (frozen_run["source"],))


@pytest.mark.parametrize("tamper", ["edit", "add", "remove"])
def test_artifact_inventory_changes_block_lock(frozen_run, tamper):
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    runner.run_stage(out, "development")
    target = out / "pilot/isolated_000/metrics.json"
    if tamper == "edit":
        runner.write_json(target, {"error": 0.0})
    elif tamper == "add":
        (out / "pilot/extra.txt").write_text("extra")
    else:
        target.unlink()
    with pytest.raises(ValueError, match="artifact hashes or inventory"):
        runner.lock_evaluation(out)
    assert not (out / "evaluation_lock.json").exists()


def test_receipt_change_after_lock_blocks_reserved_generation(frozen_run):
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    runner.run_stage(out, "development")
    runner.lock_evaluation(out)
    path = out / "pilot/receipt.json"
    data = runner.read_json(path)
    data["elapsed_seconds"] += 0.01
    runner.write_json(path, data)
    before = list(frozen_run["calls"])
    with pytest.raises(ValueError, match="Locked pilot artifacts changed"):
        runner.run_stage(out, "evaluation")
    assert frozen_run["calls"] == before


def test_measured_projection_cannot_exceed_reserved_cap(frozen_run):
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    runner.run_stage(out, "development")
    path = out / "pilot/receipt.json"
    data = runner.read_json(path)
    data["elapsed_seconds"] = 800
    runner.write_json(path, data)
    with pytest.raises(ValueError, match="projection .* exceeds reserved cap"):
        runner.lock_evaluation(out)


def test_missing_or_duplicate_declared_units_block_lock(frozen_run):
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    runner.run_stage(out, "development")
    path = out / "pilot/receipt.json"
    data = runner.read_json(path)
    data["completed_units"][-1] = data["completed_units"][0]
    runner.write_json(path, data)
    with pytest.raises(ValueError, match="missing declared units or has duplicate"):
        runner.lock_evaluation(out)


def test_timeout_retains_incomplete_receipt_and_completed_fits(frozen_run):
    out = frozen_run["output"]
    calls = 0

    def fail_second(unit, config, seed, deadline):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise TimeoutError("fixture cap")
        return {"seed": seed}

    frozen_run["apis"]["fit_peak"] = fail_second
    with pytest.raises(TimeoutError):
        runner.run_stage(out, "pilot")
    receipt = runner.read_json(out / "pilot/receipt.json")
    assert receipt["status"] == "incomplete" and receipt["completed_units"] == []
    payload = torch.load(out / "pilot/isolated_000/incomplete_payload.pt", weights_only=False)
    assert len(payload["states"]) == 1
    with pytest.raises(FileExistsError):
        runner.run_stage(out, "pilot")
    with pytest.raises(ValueError, match="not completed"):
        runner.run_stage(out, "development")


def test_failed_diagnostics_retains_explicit_error(frozen_run):
    out = frozen_run["output"]

    def broken(*args):
        raise ArithmeticError("fixture nonfinite diagnostic")

    frozen_run["apis"]["peak_diagnostics"] = broken
    with pytest.raises(ArithmeticError):
        runner.run_stage(out, "pilot")
    receipt = runner.read_json(out / "pilot/receipt.json")
    assert receipt["status"] == "error"
    assert receipt["error_type"] == "ArithmeticError"
    assert receipt["failed_unit"] == "isolated_000"


def test_gpu_contention_postpones_before_pool_access(frozen_run, monkeypatch):
    out = frozen_run["output"]
    monkeypatch.setattr(runner, "environment", lambda device: {"competing_gpu_processes": ["123, unrelated, 200"]})
    with pytest.raises(RuntimeError, match="Competing GPU work detected"):
        runner.run_stage(out, "pilot")
    assert not frozen_run["calls"]
    assert not (out / "pilot").exists()


def test_bootstrap_uses_whole_units_and_preserves_paired_effects():
    records = [{"x": 1.0, "effect": 5.0}, {"x": 3.0, "effect": 5.0}]
    result = runner.aggregate_units(records, 5000, 9)
    assert result["x"]["n_units"] == 2
    assert result["x"]["mean"] == 2
    assert result["x"]["ci95"] == [1.0, 3.0]
    assert result["effect"]["ci95"] == [5.0, 5.0]
    assert result == runner.aggregate_units(records, 5000, 9)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_no_survivor_only_aggregation(bad):
    result = runner.aggregate_units([{"metric": 1.0}, {"metric": bad}], 100, 2)["metric"]
    assert not result["complete"]
    assert result["n_finite"] == 1 and result["n_units"] == 2
    assert result["mean"] is None and result["ci95"] is None


def test_missing_metric_is_not_dropped_from_uncertainty_denominator():
    result = runner.aggregate_units([{"metric": 1.0}, {}], 100, 2)["metric"]
    assert not result["complete"] and result["n_units"] == 2


def test_json_null_records_undefined_values_without_invalid_json(tmp_path):
    path = tmp_path / "record.json"
    runner.write_json(path, {"undefined": math.nan, "number": np.float64(2), "integer": np.int64(4)})
    assert json.loads(path.read_text()) == {"undefined": None, "number": 2.0, "integer": 4}


def test_cli_returns_nonzero_on_unfrozen_run(tmp_path):
    assert runner.main(["--stage", "pilot", "--output-dir", str(tmp_path / "absent")]) == 1


def test_gpu_residency_screen_ignores_only_named_launcher_and_self(monkeypatch):
    monkeypatch.setattr(runner, "_command", lambda args: {
        "returncode": 0,
        "stdout": f"{runner.os.getpid()}, python.exe, 50\n123, C:\\Tools\\PowerToys.PowerLauncher.exe, 10\n456, python.exe, 100",
    })
    sleeps = []
    inspected = []
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    def inspect(pid):
        inspected.append(pid)
        return {"cpu_seconds": 25.0, "created_ticks": 5}

    monkeypatch.setattr(runner, "_process_cpu_snapshot", inspect)
    result = runner.gpu_preflight(torch.device("cuda"))
    assert not result["competing_gpu_processes"]
    assert inspected == [456, 456]
    assert sleeps == [5.0]
    assert len(result["gpu_processes"]) == 3
    assert "cannot prove GPU idleness" in result["gpu_cpu_check"]["limitation"]


@pytest.mark.parametrize("second", [
    {"cpu_seconds": 25.01, "created_ticks": 5},
    {"cpu_seconds": 25.0, "created_ticks": 6},
    {"error": "permission unavailable"},
])
def test_gpu_active_reused_or_uninspectable_pid_blocks(second, monkeypatch):
    monkeypatch.setattr(runner, "_gpu_processes", lambda: ({"returncode": 0}, [
        {"pid": 456, "name": "python.exe", "ignored_reason": None}]))
    readings = iter([{"cpu_seconds": 25.0, "created_ticks": 5}, second])
    monkeypatch.setattr(runner, "_process_cpu_snapshot", lambda pid: next(readings))
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    result = runner.gpu_preflight(torch.device("cuda"))
    assert len(result["competing_gpu_processes"]) == 1


@pytest.mark.parametrize("clock,expected", [(25.0, False), (25.1, True)])
def test_end_snapshot_retains_stage_cpu_comparison(clock, expected, monkeypatch):
    monkeypatch.setattr(runner, "_gpu_processes", lambda: ({"returncode": 0}, [
        {"pid": 456, "name": "python.exe", "ignored_reason": None}]))
    monkeypatch.setattr(runner, "_process_cpu_snapshot", lambda pid: {"cpu_seconds": clock, "created_ticks": 5})
    initial = {"gpu_cpu_check": {"after": {"456": {"cpu_seconds": 25.0, "created_ticks": 5}}}}
    result = runner.gpu_end_snapshot(torch.device("cuda"), initial)
    assert bool(result["competing_gpu_processes"]) is expected
    assert result["cpu_after"]["456"]["cpu_seconds"] == clock


def test_observed_contention_during_stage_is_retained_without_rejecting_complete_data(frozen_run, monkeypatch):
    out = frozen_run["output"]
    monkeypatch.setattr(runner, "gpu_end_snapshot", lambda *args: {"competing_gpu_processes": ["fixture activity"]})
    runner.run_stage(out, "pilot")
    receipt = runner.read_json(out / "pilot/receipt.json")
    assert receipt["status"] == "complete" and len(receipt["completed_units"]) == 6
    assert receipt["environment_end"]["competing_gpu_processes"]


@pytest.fixture
def postflight_run(frozen_run):
    """Represent the original operational error using only disposable fake data."""
    out = frozen_run["output"]
    runner.run_stage(out, "pilot")
    path = out / "pilot/receipt.json"
    receipt = runner.read_json(path)
    receipt.update(status="error", error_type="RuntimeError", error=runner.POSTFLIGHT_ERROR_PREFIX + "fixture activity",
                   environment_end={"competing_gpu_processes": [{"pid": 7228, "reason": "CPU clock advanced during stage"}]})
    receipt.pop("artifact_manifest_sha256")
    runner.write_json(path, receipt)
    (out / "pilot/stage_summary.json").unlink()
    (out / "pilot/artifact_manifest.json").unlink()
    original = frozen_run["runner_source"].read_bytes()
    (out / "frozen/runner_before_postflight_fix.py").write_bytes(original)
    frozen_run["runner_source"].write_text("# amended operational runner fixture\n", encoding="utf-8")
    frozen_run["original_receipt_bytes"] = path.read_bytes()
    frozen_run["original_manifest_bytes"] = (out / "frozen/source_manifest.json").read_bytes()
    return frozen_run


def test_operational_amendment_and_adjudication_preserve_originals_and_bind_lock(postflight_run):
    out = postflight_run["output"]
    unit_before = {unit: runner._inventory(out / "pilot" / unit)
                   for unit in runner.read_json(out / "pilot/receipt.json")["completed_units"]}
    before_calls = list(postflight_run["calls"])
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_freeze(out)
    amendment = runner.prepare_source_amendment(out, "Operational fixture fix; no scientific changes")
    assert amendment["old_runner_sha256"] != amendment["new_runner_sha256"]
    terminal = runner.adjudicate_completed_postflight(out)
    assert terminal["status"] == "complete_with_operational_adjudication"
    assert terminal["elapsed_seconds"] > terminal["original_elapsed_seconds"]
    assert postflight_run["calls"] == before_calls
    assert (out / "pilot/receipt.json").read_bytes() == postflight_run["original_receipt_bytes"]
    assert (out / "frozen/source_manifest.json").read_bytes() == postflight_run["original_manifest_bytes"]
    assert all(runner._inventory(out / "pilot" / unit) == original for unit, original in unit_before.items())
    config, source_hash = runner.verify_freeze(out)
    assert runner.verify_stage(out, "pilot", config, source_hash) == terminal
    development = runner.run_stage(out, "development")
    amendment_hash = runner.source_amendment_hash(out)
    assert development["source_amendment_sha256"] == amendment_hash
    lock = runner.lock_evaluation(out)
    assert lock["source_amendment_sha256"] == amendment_hash
    runner.run_stage(out, "evaluation")
    summary = runner.analyze(out)
    assert summary["source_amendment_sha256"] == amendment_hash
    with pytest.raises(FileExistsError):
        runner.adjudicate_completed_postflight(out)


def test_amendment_cannot_accept_changed_scientific_source(postflight_run):
    postflight_run["source"].write_text("# changed science\n")
    with pytest.raises(ValueError, match="Scientific source changed"):
        runner.prepare_source_amendment(postflight_run["output"], "attempt")


def test_amended_runner_drift_and_archived_original_drift_are_rejected(postflight_run):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    new_bytes = postflight_run["runner_source"].read_bytes()
    postflight_run["runner_source"].write_text("# another change\n")
    with pytest.raises(ValueError, match="operational source amendment"):
        runner.verify_freeze(out)
    postflight_run["runner_source"].write_bytes(new_bytes)
    (out / "frozen/runner_before_postflight_fix.py").write_text("# false original\n")
    with pytest.raises(ValueError, match="operational source amendment"):
        runner.verify_freeze(out)


@pytest.mark.parametrize("change", ["different_error", "missing_unit", "cap", "incomplete_fit", "missing_payload"])
def test_only_exact_complete_within_cap_postflight_error_is_adjudicable(postflight_run, change):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    path = out / "pilot/receipt.json"
    receipt = runner.read_json(path)
    if change == "different_error":
        receipt["error"] = "ordinary scientific failure"
    elif change == "missing_unit":
        receipt["completed_units"].pop()
    elif change == "cap":
        receipt["elapsed_seconds"] = postflight_run["config"]["pools"]["pilot"]["cap_seconds"]
    elif change == "incomplete_fit":
        unit_path = out / "pilot/isolated_000/receipt.json"
        unit = runner.read_json(unit_path)
        unit["fits"] = 2
        runner.write_json(unit_path, unit)
    elif change == "missing_payload":
        (out / "pilot/isolated_000/payload.pt").unlink()
    runner.write_json(path, receipt)
    with pytest.raises(ValueError):
        runner.adjudicate_completed_postflight(out)
    assert not (out / "pilot/completion_adjudication.json").exists()


def test_original_receipt_hash_is_checked_after_adjudication(postflight_run):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    runner.adjudicate_completed_postflight(out)
    path = out / "pilot/receipt.json"
    receipt = runner.read_json(path)
    receipt["elapsed_seconds"] += .001
    runner.write_json(path, receipt)
    config, source_hash = runner.verify_freeze(out)
    with pytest.raises(ValueError, match="changed original error receipt"):
        runner.verify_stage(out, "pilot", config, source_hash)


def test_reserved_lock_binds_amendment_identity(postflight_run):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    runner.adjudicate_completed_postflight(out)
    runner.run_stage(out, "development")
    runner.lock_evaluation(out)
    path = out / "source_amendment.json"
    amendment = runner.read_json(path)
    amendment["reason"] += " changed later"
    runner.write_json(path, amendment)
    before = list(postflight_run["calls"])
    with pytest.raises(ValueError, match="Reserved lock does not bind"):
        runner.run_stage(out, "evaluation")
    assert postflight_run["calls"] == before


@pytest.mark.parametrize("change", ["missing_unit", "wrong_hash"])
def test_adjudication_retained_unit_inventory_is_verified(postflight_run, change):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    runner.adjudicate_completed_postflight(out)
    path = out / "pilot/completion_adjudication.json"
    receipt = runner.read_json(path)
    prior = receipt["unit_artifacts_before_adjudication"]
    if change == "missing_unit":
        prior.pop("isolated_000")
    else:
        prior["isolated_000"]["payload.pt"] = "wrong hash"
    runner.write_json(path, receipt)
    config, source_hash = runner.verify_freeze(out)
    with pytest.raises(ValueError, match="exact original unit artifact inventory"):
        runner.verify_stage(out, "pilot", config, source_hash)


def test_adjudication_charges_terminal_verification_and_invalidates_late_completion(postflight_run, monkeypatch):
    out = postflight_run["output"]
    runner.prepare_source_amendment(out, "fixture correction")
    clock = [0.0]
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    original_verify = runner.verify_stage

    def late_verification(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        clock[0] = postflight_run["config"]["pools"]["pilot"]["cap_seconds"] + 1
        return result

    monkeypatch.setattr(runner, "verify_stage", late_verification)
    with pytest.raises(TimeoutError, match="Terminal adjudication"):
        runner.adjudicate_completed_postflight(out)
    receipt = runner.read_json(out / "pilot/completion_adjudication.json")
    assert receipt["status"] == "incomplete"
    assert receipt["elapsed_seconds"] > postflight_run["config"]["pools"]["pilot"]["cap_seconds"]
    assert (out / "pilot/receipt.json").read_bytes() == postflight_run["original_receipt_bytes"]
    config, source_hash = runner.verify_freeze(out)
    with pytest.raises(ValueError, match="Invalid completion adjudication"):
        original_verify(out, "pilot", config, source_hash)
