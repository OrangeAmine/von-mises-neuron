"""Checks that stage orchestration cannot silently change selection or evidence."""

from __future__ import annotations

import copy
import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_vm_locality as runner


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/vm_locality_v2.json").read_text())


def pilot_fixture(config):
    rows = runner.planned_runs(config, "pilot")
    summaries = []
    for cell in rows:
        # The timing panel emphatically prefers .03; it must not enter selection.
        score = (0.01 if cell["role"] == "timing" else
                 {0.01: 3., 0.03: 2., 0.1: 1., None: .5}[cell["learning_rate"]])
        summaries.append({"status": "completed", "best_validation_mse": score,
                          "primary_error": 999-score, "fit_seconds": .5,
                          "overshoot_seconds": .01})
    manifest = {"phase": "pilot", "rows": [{"cell": c, "status": "completed",
                                             "fit_call_seconds": .6, "wall_seconds": .7}
                                            for c in rows], "evidence_hashes": {"fixture": "digest"}}
    return manifest, summaries


def setup_selection(monkeypatch, config):
    manifest, summaries = pilot_fixture(config)
    monkeypatch.setattr(runner, "completed_run", lambda *args: (manifest, summaries))
    monkeypatch.setattr(runner, "source_manifest", lambda: {"test": "source"})
    monkeypatch.setattr(runner, "relative", lambda p: p.name)
    monkeypatch.setattr(runner, "file_hash", lambda p: "digest")
    return manifest, summaries


def test_pilot_selects_noisy_validation_only_and_excludes_timing(monkeypatch, tmp_path, config):
    _, summaries = setup_selection(monkeypatch, config)
    one = runner.select_pilots(tmp_path/"pilot", tmp_path/"first.json", config)
    for summary in summaries:
        summary["primary_error"] = -summary["primary_error"] * 1e10
    two = runner.select_pilots(tmp_path/"pilot", tmp_path/"second.json", config)
    assert one["learning_rates"] == two["learning_rates"] == {
        "vp_centers": .1, "vp_widths": .1, "vp_both": .1, "joint_both": .1}
    assert all(candidate["count"] == 8 for candidates in one["scores"].values() for candidate in candidates)


def test_pilot_exact_ties_choose_smaller_rate(monkeypatch, tmp_path, config):
    _, summaries = setup_selection(monkeypatch, config)
    for summary in summaries:
        summary["best_validation_mse"] = 1.
    result = runner.select_pilots(tmp_path/"pilot", tmp_path/"selection.json", config)
    assert set(result["learning_rates"].values()) == {.01}


def test_missing_or_nonfinite_pilot_candidate_rejected(monkeypatch, tmp_path, config):
    manifest, summaries = setup_selection(monkeypatch, config)
    index = next(i for i,r in enumerate(manifest["rows"]) if r["cell"]["role"] == "selection"
                 and r["cell"]["kind"] != "ridge")
    summaries[index]["best_validation_mse"] = float("nan")
    with pytest.raises(ValueError, match="Incomplete"):
        runner.select_pilots(tmp_path/"pilot", tmp_path/"bad.json", config)
    summaries.pop(index)
    manifest["rows"].pop(index)
    with pytest.raises(ValueError, match="Incomplete"):
        runner.select_pilots(tmp_path/"pilot", tmp_path/"missing.json", config)
    assert not (tmp_path/"bad.json").exists()


def test_existing_selection_never_overwritten(monkeypatch, tmp_path, config):
    setup_selection(monkeypatch, config)
    target = tmp_path/"selection.json"
    runner.select_pilots(tmp_path/"pilot", target, config)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        runner.select_pilots(tmp_path/"pilot", target, config)
    assert target.read_bytes() == original


def test_completed_run_rejects_missing_reordered_or_extra_cells(monkeypatch, tmp_path, config):
    cells = runner.planned_runs(config, "pilot")
    record = {"kind": "locality_run_v2", "phase": "pilot", "complete": True,
              "status": "completed", "deadline_respected": True,
              "rows": [{"cell": c, "status": "completed"} for c in cells]}
    monkeypatch.setattr(runner, "verify_run_manifest", lambda *args: (record, []))
    for malformed in (cells[:-1], cells[::-1], cells + cells[:1]):
        record["rows"] = [{"cell": c, "status": "completed"} for c in malformed]
        with pytest.raises(ValueError, match="matrix"):
            runner.completed_run(tmp_path/"manifest", config)


@pytest.mark.parametrize("field,value", [("complete",False), ("status","interrupted"),
                                        ("deadline_respected",False), ("kind","different")])
def test_completed_run_rejects_incomplete_invocation(monkeypatch, tmp_path, config, field, value):
    record = {"kind":"locality_run_v2","phase":"pilot","complete":True,
              "status":"completed","deadline_respected":True}
    record[field] = value
    monkeypatch.setattr(runner, "verify_run_manifest", lambda *args: (record, []))
    with pytest.raises(ValueError, match="complete"):
        runner.completed_run(tmp_path/"manifest", config)


def test_feasibility_uses_only_adaptive_atomic_overshoot(monkeypatch, tmp_path, config):
    manifest, summaries = setup_selection(monkeypatch, config)
    rates = {m["id"]: .03 for m in config["models"] if m["kind"] != "ridge"}
    monkeypatch.setattr(runner, "verify_pilot_selection", lambda *args: {
        "input_manifest": "pilot", "learning_rates": rates})
    for row, summary in zip(manifest["rows"], summaries):
        summary["overshoot_seconds"] = 50 if row["cell"]["kind"] == "ridge" else .01
    result = runner.feasibility(tmp_path/"pilot", tmp_path/"selected", tmp_path/"feasible.json", config)
    assert result["measured"]["max_overshoot"] == .01
    assert result["estimates"]["confirmation"]["adaptive"] == 1536
    assert result["all_passed"] is True


def test_interrupted_fit_is_journaled_and_stops_phase(monkeypatch, tmp_path, config):
    from src import locality_tasks, locality_training
    cells = runner.planned_runs(config, "pilot")[:2]
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "planned_runs", lambda *args: cells)
    monkeypatch.setattr(runner, "snapshot_sources", lambda *args: {})
    monkeypatch.setattr(runner, "environment", lambda: {"test_fixture": True})
    monkeypatch.setattr(locality_training, "warmup_locality", lambda **kwargs: None)
    monkeypatch.setattr(locality_tasks, "initializer", lambda *args: {})

    class TaskFixture:
        def to_dict(self):
            return {"test_fixture": True}

        def for_budget(self, *args):
            return {}

    monkeypatch.setattr(locality_tasks, "generate_locality_task", lambda *args, **kwargs: TaskFixture())
    calls = []

    def interrupted_fit(*args, **kwargs):
        calls.append(args[0])
        return {"summary": {"status": "interrupted"}, "artifacts": {}}

    monkeypatch.setattr(locality_training, "fit_locality", interrupted_fit)
    output = tmp_path/"interrupted"
    args = Namespace(phase="pilot", output=output, selection=None, feasibility=None, lock=None)
    assert runner.run_phase(args, config) == 1
    manifest = runner.read_json(output/"manifest.json")
    assert len(calls) == 1
    assert [row["status"] for row in manifest["rows"]] == ["interrupted", "pending"]
    assert manifest["complete"] is False
    record = runner.read_json(output/"fits"/cells[0]["id"]/"record.json")
    assert record["status"] == "interrupted"
    assert "wall_seconds" not in record
    assert manifest["rows"][0]["wall_seconds"] > 0
    assert manifest["rows"][0]["record_path"] in manifest["evidence_hashes"]
    assert runner.read_json(output/"progress.json")["status"] == "interrupted"
    with pytest.raises(FileExistsError):
        runner.run_phase(args, config)
    assert len(calls) == 1
