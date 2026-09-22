"""Run registration, no-survivor selection, timeout and resumption contracts."""

import copy
import json

import pytest
import numpy as np
import torch

import scripts.run_vm_methodology as runner


@pytest.fixture
def config():
    return runner.read_json(runner.CONFIG_PATH)


@pytest.fixture
def mocked_execution(monkeypatch):
    monkeypatch.setattr(runner,"snapshot_sources",lambda output:{"fixture":"hash"})
    monkeypatch.setattr(runner,"source_manifest",lambda:{"fixture":"hash"})
    monkeypatch.setattr(runner,"environment",lambda:{"test":"mocked execution"})
    calls = []
    def fit(row,task,config,deadline,device):
        calls.append(row["run_id"])
        return {"summary":{"status":"completed","validation_curve_area_normalized":row.get("learning_rate",0),
                            "best_validation_mse":row.get("learning_rate",0)},"artifacts":{}}
    monkeypatch.setattr(runner,"_fit",fit)
    return calls


def test_records_complete_matrix_and_never_repeats_completed(config,mocked_execution,tmp_path):
    directory = tmp_path/"pilot"
    first = runner.execute_stage(config,"A","pilot",directory)
    assert first["status"] == "completed"
    assert len(mocked_execution) == 52
    assert len(first["tasks"]) == 2
    runner.execute_stage(config,"A","pilot",directory,resume=True)
    assert len(mocked_execution) == 52
    with pytest.raises(FileExistsError):
        runner.execute_stage(config,"A","pilot",directory)


def test_failure_is_retained_and_not_retried(config,mocked_execution,monkeypatch,tmp_path):
    def fail(*args,**kwargs):
        raise RuntimeError("deliberate disposable test failure")
    monkeypatch.setattr(runner,"_fit",fail)
    directory = tmp_path/"pilot"
    result = runner.execute_stage(config,"A","pilot",directory)
    assert result["status"] == "incomplete"
    assert all(row["status"] == "failed" for row in result["runs"])
    monkeypatch.setattr(runner,"_fit",lambda *a,**k:pytest.fail("Failed fit retried"))
    runner.execute_stage(config,"A","pilot",directory,resume=True)


def test_confirmation_guard_precedes_generation(config,monkeypatch,tmp_path):
    import src.methodology_tasks as tasks
    monkeypatch.setattr(tasks,"generate_task",lambda *a,**k:pytest.fail("Reserved generation reached"))
    with pytest.raises(ValueError,match="lock"):
        runner.execute_stage(config,"A","confirmation",tmp_path/"forbidden")
    assert not (tmp_path/"forbidden").exists()


def test_timeout_retains_full_pending_matrix(config,mocked_execution,tmp_path):
    config = copy.deepcopy(config)
    config["execution"]["timeouts_minutes"]["A_pilot"] = 0
    result = runner.execute_stage(config,"A","pilot",tmp_path/"timeout")
    assert result["status"] == "incomplete"
    assert len(result["runs"]) == 52
    assert {row["status"] for row in result["runs"]} == {"not_run_timeout"}
    assert mocked_execution == []


def test_incomplete_rate_cannot_win_on_survivors(config,tmp_path,mocked_execution):
    directories = []
    for stage in ("A","B"):
        directory = tmp_path/stage
        directories.append(directory)
        rows = runner.planned_runs(config,stage,"pilot")
        for row in rows:
            row["status"] = "completed"
            row["summary"] = {"validation_curve_area_normalized":row.get("learning_rate",1),
                              "best_validation_mse":row.get("learning_rate",1)}
        if stage == "A":
            next(row for row in rows if row.get("optimizer") == "Adam" and row.get("learning_rate") == .001)["status"] = "failed"
        runner.write_json(directory/"manifest.json",{"phase":"pilot","stage":stage,"config_hash":runner.object_hash(config),"runs":rows,"source_hashes":{"fixture":"hash"}})
    selected = runner.select_pilots(config,directories)
    assert selected["pilot_hyperparameters"]["A"]["Adam"] == .01
    assert selected["pilot_hyperparameters"]["A"]["SGD"] == .003


def test_initialization_artifacts_allow_restricted_loading(tmp_path):
    value = {"weights":np.ones((2,3)),"metadata":{"seed":np.int64(31),"components":[1,2]}}
    path = tmp_path/"initialization.pt"
    torch.save(runner._tensor_artifact(value),path)
    loaded = torch.load(path,weights_only=True)
    assert loaded["metadata"]["seed"] == 31
    torch.testing.assert_close(loaded["weights"],torch.ones(2,3,dtype=torch.float64))


def test_interruption_saves_partial_bundle_before_stopping(config,mocked_execution,monkeypatch,tmp_path):
    def interrupted(*a,**k):
        return {"summary":{"status":"interrupted","history":[{"step":0}]},
                "artifacts":{"initial_state":{"bias":torch.zeros(1)}}}
    monkeypatch.setattr(runner,"_fit",interrupted)
    directory = tmp_path/"interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.execute_stage(config,"A","pilot",directory)
    manifest = runner.read_json(directory/"manifest.json")
    assert manifest["runs"][0]["status"] == "interrupted"
    assert (directory/"fits"/manifest["runs"][0]["run_id"]/"artifacts.pt").exists()
    assert all(row["status"] == "pending" for row in manifest["runs"][1:])


def test_resume_never_renews_absolute_deadline(config,mocked_execution,tmp_path):
    directory = tmp_path/"expired"
    runner.execute_stage(config,"A","pilot",directory)
    path = directory/"manifest.json"
    manifest = runner.read_json(path)
    # Model an abrupt stop before the last scheduled call, with stale elapsed
    # progress but an expired original invocation deadline.
    manifest["runs"][-1]["status"] = "pending"
    manifest["time_seconds"] = 0
    manifest["deadline_unix"] = 0
    runner.write_json(path,manifest,exclusive=False)
    count = len(mocked_execution)
    resumed = runner.execute_stage(config,"A","pilot",directory,resume=True)
    assert len(mocked_execution) == count
    assert resumed["runs"][-1]["status"] == "not_run_timeout"


def test_prelock_observer_repairs_never_allow_training_source_changes():
    original = {"scripts/run_vm_methodology.py":"a","scripts/analyze_vm_methodology.py":"a",
                "src/methodology_training.py":"a","src/methodology_tasks.py":"a",
                "configs/vm_methodology_v1.json":"a"}
    changed = {**original,"scripts/analyze_vm_methodology.py":"b"}
    assert runner._pilot_training_sources_match(original,changed)
    for name in ("src/methodology_training.py","src/methodology_tasks.py","configs/vm_methodology_v1.json"):
        assert not runner._pilot_training_sources_match(original,{**changed,name:"b"})
    assert not runner._pilot_training_sources_match(original,{"scripts/analyze_vm_methodology.py":"a"})


def test_fixed_composition_timing_uses_validation_choices_not_fastest_runs(config,mocked_execution,tmp_path):
    directories = []
    for stage in ("A","B"):
        directory = tmp_path/stage
        directories.append(directory)
        rows = runner.planned_runs(config,stage,"pilot")
        for row in rows:
            row["status"] = "completed"
            # The validation winner is deliberately slower. Timing may not
            # select the cheap candidate or omit its slow family observation.
            selected = row.get("learning_rate") in (.1,.3,.01) if stage == "A" else row.get("learning_rate") == .01
            row["wall_seconds"] = 2. if selected else .1
            row["summary"] = {"validation_curve_area_normalized":-row.get("learning_rate",0),
                              "best_validation_mse":-row.get("learning_rate",0)}
        runner.write_json(directory/"manifest.json",{"phase":"pilot","stage":stage,
            "config_hash":runner.object_hash(config),"runs":rows,"source_hashes":{"fixture":"hash"},
            "time_seconds":sum(row["wall_seconds"] for row in rows)})
    result = runner.pilot_feasibility(config,directories)
    row = next(item for item in result["projections"] if item["stage"] == "B" and item["phase"] == "confirmation")
    assert all(item["validation_selected_learning_rate"] == .01 for item in row["fixed_composition_strata"])
    assert all(item["all_pilot_fit_seconds"] == [2.]*4 for item in row["fixed_composition_strata"])
    assert sum(item["planned_fits"] for item in row["fixed_composition_strata"]) == 576
    assert sum(item["projected_p90_seconds"] for item in row["fixed_composition_strata"]) == 576*6*2
    assert row["fits_ceiling"] is False


def _detailed_stub_fit(row, task, config, deadline, device):
    """Produce realistic nested evidence without any model/optimizer execution."""
    summary = {"status":"completed", "history":[{"step":i,"validation_mse":1/(i+1)} for i in range(30)],
               "update_history":[{"step":i,"objective":i/100,"parameter_gradient_norms":{"bias":.3}} for i in range(120)],
               "validation_curve_area_normalized":.4, "best_validation_mse":.2,
               "diagnostics":{"unchanged":[True,None,1.25]},"run_id":row["run_id"]}
    return {"summary":summary,"artifacts":{"initial_state":{"bias":torch.zeros(1)},
            "final_state":{"bias":torch.ones(1)},"selected_state":{"bias":torch.tensor([.5])},
            "predictions":{"grid":torch.arange(7,dtype=torch.float64).reshape(-1,1)}}}


def test_progress_references_preserve_final_full_summaries_and_tensor_artifacts(config,mocked_execution,monkeypatch,tmp_path):
    monkeypatch.setattr(runner,"_fit",_detailed_stub_fit)
    original_write = runner.write_json
    progress_checks = []

    def observe_write(path,value,**kwargs):
        if path.name == "manifest.json" and value.get("status") == "running":
            assert value["progress_representation"] == "lightweight_summary_references_v1"
            for row in value["runs"]:
                if row["status"] == "completed":
                    assert "summary" not in row
                    assert row["summary_ref"] == f"fits/{row['run_id']}/summary.json"
                    assert row["summary_ref"] in row["artifact_hashes"]
            progress_checks.append(sum(row["status"] == "completed" for row in value["runs"]))
        original_write(path,value,**kwargs)

    monkeypatch.setattr(runner,"write_json",observe_write)
    directory = tmp_path/"lossless"
    result = runner.execute_stage(config,"A","pilot",directory)
    assert len(result["runs"]) == 52
    assert max(progress_checks) == 52
    final = runner.read_json(directory/"manifest.json")
    assert final == result
    assert final["progress_representation"] == "full_summaries_v1"
    assert final["status"] == "completed"
    for row in final["runs"]:
        assert "summary_ref" not in row
        expected = _detailed_stub_fit(row,None,None,None,None)
        assert row["summary"] == expected["summary"]
        assert runner.read_json(directory/f"fits/{row['run_id']}/summary.json") == expected["summary"]
        saved = torch.load(directory/f"fits/{row['run_id']}/artifacts.pt",weights_only=True)
        for state in ("initial_state","final_state","selected_state"):
            assert torch.equal(saved[state]["bias"],expected["artifacts"][state]["bias"])
        assert torch.equal(saved["predictions"]["grid"],expected["artifacts"]["predictions"]["grid"])


def test_hard_stop_lightweight_resume_hydrates_without_repeating_complete_or_running_rows(config,mocked_execution,monkeypatch,tmp_path):
    original_write = runner.write_json
    crash_payload = None
    calls = []

    def fit(row,*args):
        calls.append(row["run_id"])
        return _detailed_stub_fit(row,*args)

    def crash_after_registration(path,value,**kwargs):
        nonlocal crash_payload
        original_write(path,value,**kwargs)
        if (path.name == "manifest.json" and value.get("status") == "running"
                and sum(row["status"] == "completed" for row in value["runs"]) == 1
                and sum(row["status"] == "running" for row in value["runs"]) == 1):
            crash_payload = copy.deepcopy(value)
            raise SystemExit("simulate process stop before second fit")

    monkeypatch.setattr(runner,"_fit",fit)
    monkeypatch.setattr(runner,"write_json",crash_after_registration)
    directory = tmp_path/"crashed"
    with pytest.raises(SystemExit):
        runner.execute_stage(config,"A","pilot",directory)
    assert runner.read_json(directory/"manifest.json")["progress_representation"] == "full_summaries_v1"
    assert len(calls) == 1 and crash_payload is not None
    completed_id = crash_payload["runs"][0]["run_id"]
    interrupted_id = crash_payload["runs"][1]["run_id"]
    first_hashes = copy.deepcopy(crash_payload["runs"][0]["artifact_hashes"])
    # Restore the last atomic progress checkpoint, representing an actual hard
    # kill where graceful finalization could not have executed.
    original_write(directory/"manifest.json",crash_payload,exclusive=False)
    monkeypatch.setattr(runner,"write_json",original_write)
    resumed = runner.execute_stage(config,"A","pilot",directory,resume=True)
    assert resumed["status"] == "incomplete"
    assert len(calls) == 51
    assert calls.count(completed_id) == 1
    assert interrupted_id not in calls
    assert resumed["runs"][1]["status"] == "interrupted"
    assert resumed["runs"][0]["artifact_hashes"] == first_hashes
    assert resumed["runs"][0]["summary"] == _detailed_stub_fit(resumed["runs"][0],None,None,None,None)["summary"]
    assert resumed["progress_representation"] == "full_summaries_v1"
    assert all("summary_ref" not in row for row in resumed["runs"])


@pytest.mark.parametrize("corruption",["bytes","missing_file","reference","missing_reference","hash"])
def test_corrupt_lightweight_progress_rejected_before_any_fit(config,mocked_execution,monkeypatch,tmp_path,corruption):
    directory = tmp_path/"corrupt"
    result = runner.execute_stage(config,"A","pilot",directory)
    result["status"] = "running"
    payload = runner.progress_manifest_payload(result)
    row = payload["runs"][0]
    summary_path = directory/row["summary_ref"]
    if corruption == "bytes":
        summary_path.write_text("{}",encoding="utf-8")
    elif corruption == "missing_file":
        summary_path.rename(summary_path.with_suffix(".preserved"))
    elif corruption == "reference":
        row["summary_ref"] = "../outside.json"
    elif corruption == "missing_reference":
        del row["summary_ref"]
    else:
        row["artifact_hashes"][row["summary_ref"]] = "incorrect"
    runner.write_json(directory/"manifest.json",payload,exclusive=False)
    monkeypatch.setattr(runner,"_fit",lambda *a,**k:pytest.fail("Corrupt progress reached a fit"))
    with pytest.raises(ValueError,match="summary|artifact|reference"):
        runner.execute_stage(config,"A","pilot",directory,resume=True)


def test_unregistered_partial_summary_and_failures_are_not_dropped():
    row = {"run_id":"partial","status":"interrupted","error":"partial save",
           "summary":{"status":"completed","history":[{"step":0}]},"custom_flag":True}
    manifest = {"status":"running","runs":[row,{"run_id":"failed","status":"failed","error":"retained"}]}
    before = copy.deepcopy(manifest)
    payload = runner.progress_manifest_payload(manifest)
    assert payload["runs"] == manifest["runs"]
    assert manifest == before
    manifest["status"] = "interrupted"
    assert runner.progress_manifest_payload(manifest)["runs"] == manifest["runs"]


def test_lossless_progress_roundtrip_restores_unknown_flags_and_full_arrays(tmp_path):
    row = {"run_id":"record","status":"completed","planned":True,
           "wall_seconds":.2,"extra_flags":[False,None,"retained"],
           "summary":{"status":"completed","history":[{"step":0,"loss":.125}],"arbitrary":[[1,2],[3,4]]}}
    path = tmp_path/"fits/record/summary.json"
    runner.write_json(path,row["summary"])
    row["artifact_hashes"] = {"fits/record/summary.json":runner.file_hash(path)}
    original = {"status":"running","runs":[row],"custom_top_level":{"seed":22}}
    payload = runner.progress_manifest_payload(original)
    hydrated = runner.hydrate_progress_manifest(json.loads(json.dumps(payload)),tmp_path)
    assert hydrated["runs"] == original["runs"]
    assert hydrated["custom_top_level"] == original["custom_top_level"]
