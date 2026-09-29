"""Failure-sensitive source guards and VM-N01 runner bookkeeping checks."""

import copy
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from src.smooth_local_protocol_v1 import (
    contained_path, digest, file_hash, load_config, verify_lock, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_vm_n01", ROOT / "scripts/run_vm_n01.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_planned_counts_and_no_extra_tuning():
    config = load_config()
    rates = {s["id"]: .01 for s in config["models"] if s["kind"] != "ridge"}
    for phase, count in config["phases"]["planned_fits"].items():
        matrix = RUNNER.planned(config, phase, rates)
        assert len(matrix) == count
        assert len({r["fit_id"] for r in matrix}) == count
        assert {r["labels"] for r in matrix} == ({256} if phase == "pilot" else {64, 256})
    pilot = RUNNER.planned(config, "pilot", rates)
    assert sum(r["model_id"] == "hybrid_joint" for r in pilot) == 10
    assert sum(r["model_id"] == "fourier16" for r in pilot) == 5


def test_failed_or_late_selection_is_retained():
    config = load_config()
    rows = RUNNER.rows_from_bundle({"fit_id": "bad"}, config["models"][0], {
        "status": "completed", "selected": {"0.25": None, "1": 0},
        "checkpoints": [{"elapsed": .5, "step": 0, "validation_mse": 2.}],
        "metrics": {"1": {"normalized_mse": 3.}}, "fit_seconds": .5}, config)
    assert len(rows) == 2
    assert rows[0]["status"] == "no_eligible_checkpoint"
    assert rows[0]["normalized_mse"] is None
    assert rows[1]["normalized_mse"] == 3.
    failed = RUNNER.rows_from_bundle({}, config["models"][0], {"status": "failed"}, config)
    assert all(r["status"] == "failed" for r in failed)


def test_projection_uses_full_caps_and_measured_overhead():
    config = load_config()
    pilot = {"selected_rates": {s["id"]: .01 for s in config["models"] if s["kind"] != "ridge"},
        "timing": {"maximum_adaptive_overshoot": .1, "maximum_fixed_fit_seconds": .2, "maximum_nonfit_seconds": .05}}
    value = RUNNER.feasibility(config, pilot, "confirmation", 7.)
    assert value["adaptive_fits"] == 480
    assert value["fixed_fits"] == 840
    assert value["projected_seconds"] == pytest.approx(480*1.1+840*.2+1320*.05+60+7)
    pilot["timing"]["maximum_nonfit_seconds"] = 2.
    assert not RUNNER.feasibility(config, pilot, "confirmation")["passed"]


def make_lock(tmp_path):
    from src.smooth_local_analysis_v1 import select_learning_rates, select_models

    config = copy.deepcopy(load_config())
    config["tasks"].update(families=["smooth"], periodic_families=["smooth"],
        replicates={"pilot": 1, "development": 1, "confirmation": 1},
        initializations={"pilot": 1, "development": 1, "confirmation": 1},
        pilot_label_budgets=[64], label_budgets=[64])
    config["models"] = [row for row in config["models"] if row["id"] in
                         {"hybrid_joint", "hybrid_vp", "fourier36", "fixed36", "hybrid_fixed36"}]
    config["phases"]["planned_fits"].update(pilot=7, development=5)
    config["phases"]["ceiling_seconds"] = {phase: 10 for phase in ("pilot", "development", "confirmation")}
    names = ["src/smooth_local_v1.py", "src/smooth_local_training_v1.py", "src/smooth_local_tasks_v1.py",
        "src/smooth_local_protocol_v1.py", "src/smooth_local_analysis_v1.py", "src/smooth_local_forecaster_v1.py",
        "src/locality_models.py", "src/circular_layers_v2.py", "scripts/run_vm_n01.py", "scripts/report_vm_n01.py",
        "scripts/validate_smooth_local_forecaster_v1.py", "scripts/benchmark_standard_dataset.py",
        "configs/vm_n01_v1.json", "docs/vm_n01_protocol.md", "docs/vm_n01_protocol_freeze.json"]
    sources = {}
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("frozen fixture")
        sources[name] = file_hash(target)
    write_json(tmp_path / "checks.json", {"full_tests": True, "standard_adapter_gate": True, "independent_review": True})
    evidence = {"checks.json": file_hash(tmp_path / "checks.json")}
    source_lock = {"kind": "vm_n01_source_lock_v1", "config_sha256": digest(config), "sources": sources,
                   "checks": dict(evidence)}
    write_json(tmp_path / "source_lock.json", source_lock)
    evidence["source_lock.json"] = file_hash(tmp_path / "source_lock.json")
    rates, models = {}, {}
    for phase in ("pilot", "development"):
        directory = tmp_path / phase
        matrix = RUNNER.planned(config, phase, rates)
        rows = []
        write_json(directory / "plan.json", {"phase": phase, "config_sha256": digest(config),
            "source_lock_sha256": file_hash(tmp_path / "source_lock.json"), "matrix": matrix})
        for item in matrix:
            artifact = directory / f"{item['fit_id']}.pt"
            artifact.write_bytes(b"fixture state, never loaded or used to generate targets")
            fit_rows = [dict(item, allowance=value, status="completed", normalized_mse=1.,
                            noisy_validation_mse=1., selected_seconds=.01)
                        for value in config["training"]["time_budgets_seconds"]]
            rows.extend(fit_rows)
            write_json(directory / f"{item['fit_id']}.json", dict(item, rows=fit_rows, artifact_sha256=file_hash(artifact)))
        if phase == "pilot":
            rates = select_learning_rates(rows, config)
        else:
            models = select_models(rows, config)
        summary = {"phase": phase, "config_sha256": digest(config), "sources": sources,
                   "complete": True, "interrupted": None, "planned_count": len(matrix),
                   "attempted_count": len(matrix), "rows": rows, "selected_rates": rates}
        if models:
            summary["selected_models"] = models
        write_json(directory / "summary.json", summary)
        write_json(directory / "receipt.json", {"complete": True, "within_ceiling": True,
            "elapsed_seconds": 1., "ceiling_seconds": 10,
            "summary_sha256": file_hash(directory / "summary.json")})
        evidence.update({p.relative_to(tmp_path).as_posix(): file_hash(p) for p in directory.iterdir()})
    receipt = {"kind": "vm_n01_confirmation_lock_v1", "config_sha256": digest(config),
        "sources": sources, "evidence": evidence, "source_lock_path": "source_lock.json",
        "phase_directories": {"pilot": "pilot", "development": "development"},
        "selected_rates": rates, "selected_models": models,
        "feasibility": {"phase": "confirmation", "projected_seconds": 5, "ceiling_seconds": 10, "passed": True},
        "prerequisites": {k: True for k in ("full_tests", "standard_adapter_gate", "independent_review", "complete_development", "complete_pilot", "feasibility")}}
    path = tmp_path / "lock.json"
    write_json(path, receipt)
    return config, path, receipt


def test_lock_verifies_all_evidence_before_reserved_access(tmp_path):
    config, path, receipt = make_lock(tmp_path)
    result = verify_lock(path, config, root=tmp_path)
    assert result["lock_sha256"] == file_hash(path)
    (tmp_path / "checks.json").write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_lock(path, config, root=tmp_path)


@pytest.mark.parametrize("mutation", ["missing_check", "wrong_config", "ceiling", "escape", "empty_sources",
    "empty_evidence", "missing_source_lock", "missing_phase", "unbound_proof", "wrong_rates", "wrong_models"])
def test_lock_rejects_invalid_prerequisites(tmp_path, mutation):
    config, path, receipt = make_lock(tmp_path)
    if mutation == "missing_check":
        receipt["prerequisites"].pop("full_tests")
    elif mutation == "wrong_config":
        config["changed"] = True
    elif mutation == "ceiling":
        receipt["feasibility"]["projected_seconds"] = 11
    elif mutation == "escape":
        receipt["sources"] = {"../outside": "hash"}
    elif mutation == "empty_sources":
        receipt["sources"] = {}
    elif mutation == "empty_evidence":
        receipt["evidence"] = {}
    elif mutation == "missing_source_lock":
        receipt.pop("source_lock_path")
    elif mutation == "missing_phase":
        receipt["phase_directories"].pop("pilot")
    elif mutation == "unbound_proof":
        receipt["evidence"].pop("pilot/plan.json")
    elif mutation == "wrong_rates":
        receipt["selected_rates"]["hybrid_joint"] = .05
    elif mutation == "wrong_models":
        receipt["selected_models"]["selected_hybrid"] = "hybrid_vp"
    write_json(path, receipt, exclusive=False)
    with pytest.raises(ValueError):
        verify_lock(path, config, root=tmp_path)


@pytest.mark.parametrize("mutation", ["wrong_stage", "wrong_source", "missing_fit", "incomplete",
                                      "wrong_matrix", "stale_receipt", "invalid_metric", "wrong_selection"])
def test_lock_rejects_internally_rehashed_invalid_phase_proof(tmp_path, mutation):
    config, path, receipt = make_lock(tmp_path)
    target = tmp_path / "pilot/summary.json"
    value = json.loads(target.read_text())
    if mutation == "wrong_stage":
        value["phase"] = "development"
    elif mutation == "wrong_source":
        value["sources"] = {}
    elif mutation == "incomplete":
        value["complete"] = False
    elif mutation == "wrong_selection":
        value["selected_rates"]["hybrid_joint"] = .05
    elif mutation == "invalid_metric":
        value["rows"][0]["status"] = "failed"
    elif mutation == "wrong_matrix":
        target = tmp_path / "pilot/plan.json"
        value = json.loads(target.read_text())
        value["matrix"] = []
    elif mutation == "missing_fit":
        receipt["evidence"].pop("pilot/fit_00000.json")
    elif mutation == "stale_receipt":
        target = tmp_path / "pilot/receipt.json"
        value = json.loads(target.read_text())
        value["summary_sha256"] = "wrong"
    write_json(target, value, exclusive=False)
    receipt["evidence"][target.relative_to(tmp_path).as_posix()] = file_hash(target)
    if target.name == "summary.json":
        terminal_path = tmp_path / "pilot/receipt.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["summary_sha256"] = file_hash(target)
        write_json(terminal_path, terminal, exclusive=False)
        receipt["evidence"]["pilot/receipt.json"] = file_hash(terminal_path)
    write_json(path, receipt, exclusive=False)
    with pytest.raises(ValueError):
        verify_lock(path, config, root=tmp_path)


def test_evidence_writes_refuse_overwrite_and_escape(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        write_json(path, {})
    with pytest.raises(ValueError):
        contained_path(tmp_path, "../sibling")
    with pytest.raises(ValueError):
        contained_path(tmp_path, ".git/config")


def test_preflight_timing_proof_cannot_authorize_reserved_access(tmp_path):
    config, path, receipt = make_lock(tmp_path)
    receipt["preflight_only"] = True
    write_json(path, receipt, exclusive=False)
    with pytest.raises(ValueError, match="Preflight"):
        verify_lock(path, config, root=tmp_path)
    assert verify_lock(path, config, root=tmp_path, allow_preflight=True)["preflight_only"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., None])
def test_complete_row_rejects_nonfinite_missing_or_negative_metrics(value):
    assert not RUNNER.complete_row({"status": "completed", "normalized_mse": value,
                                    "noisy_validation_mse": 1.})


@pytest.mark.parametrize("failure", ["task_interrupt", "final_source_mismatch"])
def test_phase_retains_terminal_receipt_for_outer_failures(tmp_path, monkeypatch, failure):
    import src.smooth_local_tasks_v1 as tasks
    import src.smooth_local_training_v1 as training
    import src.smooth_local_analysis_v1 as analysis

    config = load_config()
    source_lock = tmp_path / "source_lock.json"
    source_lock.write_text("mock source lock; no real target generation")
    source = {"sources": {"source.py": "mock hash"}}
    item = {"fit_id": "fit_00000", "phase": "pilot", "family": "smooth", "replicate": 0,
            "labels": 256, "initialization": 0, "model_id": "fourier16", "learning_rate": None}
    monkeypatch.setattr(RUNNER, "ROOT", tmp_path)
    monkeypatch.setattr(RUNNER, "verify_source_lock", lambda *args, **kwargs: source)
    monkeypatch.setattr(RUNNER, "planned", lambda *args, **kwargs: [item])
    monkeypatch.setattr(RUNNER, "environment", lambda: {})
    monkeypatch.setattr(RUNNER.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(tasks, "cell_arrays", lambda *args: {})
    monkeypatch.setattr(tasks, "initializer_rotation", lambda *args: 0.)
    monkeypatch.setattr(analysis, "select_learning_rates", lambda *args: {})
    bundle = {"status": "completed", "selected": {"0.25": 0, "1": 0},
              "checkpoints": [{"elapsed": .01, "step": 0, "validation_mse": 1.}],
              "metrics": {key: {"normalized_mse": 1.} for key in ("0.25", "1")},
              "fit_seconds": .01, "total_seconds": .02, "max_block_seconds": .01,
              "overshoot_seconds": 0., "failures": []}
    monkeypatch.setattr(training, "fit_smooth_local", lambda *args, **kwargs: copy.deepcopy(bundle))

    def generate(*args, **kwargs):
        if failure == "task_interrupt":
            raise KeyboardInterrupt("injected before any real task generation")
        return {"arrays": {}, "metadata": {}}

    def final_check(*args, **kwargs):
        if failure == "final_source_mismatch":
            raise ValueError("injected final source mismatch")

    monkeypatch.setattr(tasks, "generate_task", generate)
    monkeypatch.setattr(RUNNER, "verify_inventory", final_check)
    output = tmp_path / "results/vm_n01_v1/failure_fixture"
    args = argparse.Namespace(command="pilot", source_lock=source_lock, pilot_dir=None,
                              confirmation_lock=None, output_dir=output)
    with pytest.raises(SystemExit):
        RUNNER.run_phase(args, config)
    summary = json.loads((output / "summary.json").read_text())
    terminal = json.loads((output / "receipt.json").read_text())
    assert summary["complete"] is False and terminal["complete"] is False
    assert summary["attempted_count"] == (0 if failure == "task_interrupt" else 1)
    assert terminal["summary_sha256"] == file_hash(output / "summary.json")
    expected = "KeyboardInterrupt" if failure == "task_interrupt" else "final_source_check"
    assert expected in summary["interrupted"]
