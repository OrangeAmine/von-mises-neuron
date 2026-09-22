"""Independent observer tests on manufactured states, never formal fits/targets."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.analyze_vm_locality as audit


@pytest.fixture
def config():
    value=audit.read_json(audit.CONFIG_PATH)
    value["training"].update(max_updates=20,validation_every=10)
    value["tasks"].update(grid_points=64,label_budgets=[64])
    return value


def _arrays():
    arrays={}
    for split,count,offset in (("train",48,.1),("validation",16,.3),("grid",64,0.)):
        x=(-math.pi+(np.arange(count)+.5)*2*math.pi/count+offset)[:,None]
        clean=np.sin(x)+.2*np.cos(3*x)
        arrays[split+"_x"]=x
        if split=="grid":
            arrays["grid_clean"]=clean
            arrays["grid_noisy"]=clean+.05*np.cos(13*x)
        else:
            arrays[split+"_y"]=clean+.05*np.cos(13*x)
    return arrays


def _fixture(config,model_id="vp_both",*,arrays=None,cell=None):
    """Create mathematically consistent checkpoints without invoking a trainer."""
    arrays=_arrays() if arrays is None else arrays
    spec=next(item for item in config["models"] if item["id"]==model_id)
    cell=cell or {"phase":"pilot","role":"selection","family":"smooth","replicate":0,
        "noise_std":.05,"total_labels":64,"initialization":0,"model_id":model_id,
        "kind":spec["kind"],"learning_rate":None if spec["kind"]=="ridge" else .03,"id":"fixture"}
    count=spec["features"];kind=spec["kind"];geometry=spec.get("geometry","fixed")
    optimized=0 if kind=="ridge" else count*(2 if geometry=="both" else 1)+(count+1 if kind=="joint" else 0)
    caps=config["training"]["time_budgets_seconds"]
    steps=[0] if kind=="ridge" else [0,10,20]
    times={0:.1,10:.5,20:2.1};starts={0:.08,10:.49,20:2.09}
    history=[];states={};predictions={}
    for step in steps:
        state={"model_id":model_id,"kind":kind,"basis_type":"fourier" if model_id.startswith("fourier") else "circular",
            "basis_count":count,"fitted_parameter_count":spec["parameters"],"optimized_parameter_count":optimized,
            "readout_matches_geometry":True,"basis_state":{}}
        if state["basis_type"]=="circular":
            geo=audit._initial_geometry(config,cell)
            centers=geo["centers"]+(.001*step if geometry in ("centers","both") else 0.)
            kappas=geo["kappas"]+(.002*step if geometry in ("widths","both") else 0.)
            state.update(centers=torch.tensor(centers),kappas=torch.tensor(kappas))
            state["basis_state"]["bank.centers"]=torch.tensor(centers[:,None])
            width_key="bank.raw_kappa" if geometry in ("widths","both") else "bank.fixed_kappa"
            state["basis_state"][width_key]=torch.tensor(kappas+np.log(-np.expm1(-kappas)) if width_key.endswith("raw_kappa") else kappas)
        features=audit.numpy_features(state,arrays["train_x"])
        weight,bias=audit.independent_ridge(features,arrays["train_y"],config["training"]["penalty"])
        # Joint checkpoints intentionally are not conditional optima after zero.
        if kind=="joint" and step:
            weight=weight+.013*step
            bias=bias-.007*step
        state.update(readout_weight=torch.tensor(weight),readout_bias=torch.tensor(bias))
        predicted={split:torch.tensor(audit.numpy_prediction(state,arrays[split+"_x"])) for split in ("train","validation")}
        mse={split:float(np.mean((predicted[split].numpy()-arrays[split+"_y"])**2)) for split in predicted}
        centered=features-features.mean(axis=0)
        condition=float(np.linalg.cond(centered.T@centered+len(features)*config["training"]["penalty"]*np.eye(count)))
        history.append({"step":step,"checkpoint_start_elapsed_seconds":starts[step],"elapsed_seconds":times[step],
            "admissible_time_budgets":[cap for cap in caps if times[step]<=cap],
            "train_mse":mse["train"],"validation_mse":mse["validation"],
            "objective":.5*mse["train"]+.5*config["training"]["penalty"]*float(np.sum(weight**2)),
            "readout_diagnostics":{"normal_matrix_condition_2":condition}})
        states[str(step)]=state;predictions[str(step)]=predicted
    updates=[]
    for step in range(1,steps[-1]+1):
        time=.1+.038*step if step<=10 else .5+.158*(step-10)
        updates.append({"step":step,"elapsed_seconds":time,"objective":.11,"training_mse":.2,
            "readout_penalty":.01,"gradient_norm_before_clip":.1,"geometry_gradient_norm":.08,
            "parameter_gradient_norms":{"basis.bank.centers":.08}})
    summary={"model_id":model_id,"kind":kind,"status":"completed","penalty":config["training"]["penalty"],
        "dtype":"float64","basis_count":count,"parameter_count":spec["parameters"],
        "max_updates":config["training"]["max_updates"],"validation_every":10,"time_budgets_seconds":caps,
        "primary_time_seconds":2.,"learning_rate":cell["learning_rate"],"n_train":len(arrays["train_x"]),
        "n_validation":len(arrays["validation_x"]),"ridge_alpha":len(arrays["train_x"])*config["training"]["penalty"],
        "history":history,"update_history":updates,"updates_completed":steps[-1],
        "fit_elapsed_seconds":times[steps[-1]],"fit_seconds":times[steps[-1]],
        "overshoot_seconds":max(0.,times[steps[-1]]-max(caps)),
        "stop_reason":"fixed_closed_form" if kind=="ridge" else "wall_time","selections":{}}
    artifacts={"checkpoint_states":states,"checkpoint_predictions":predictions,"initial_state":copy.deepcopy(states["0"]),
        "final_state":copy.deepcopy(states[str(steps[-1])]),"selected_states":{},"selected_predictions":{}}
    for cap in caps:
        eligible=[row for row in history if row["elapsed_seconds"]<=cap]
        chosen=min(eligible,key=lambda row:(row["validation_mse"],row["step"]));key=str(float(cap))
        state=copy.deepcopy(states[str(chosen["step"])])
        prediction=torch.tensor(audit.numpy_prediction(state,arrays["grid_x"]))
        metrics=audit._metrics(prediction.numpy(),arrays)
        summary["selections"][key]={"allowance_seconds":cap,"eligible_checkpoints":len(eligible),
            "actual_trajectory_fit_seconds":times[steps[-1]],"completed_updates_within_allowance":sum(row["elapsed_seconds"]<=cap for row in updates),
            "status":"completed","selected_step":chosen["step"],"best_validation_mse":chosen["validation_mse"],
            "checkpoint_elapsed_seconds":chosen["elapsed_seconds"],"last_admissible_checkpoint_step":eligible[-1]["step"],
            "last_admissible_checkpoint_seconds":eligible[-1]["elapsed_seconds"],"checkpoint_budget_utilization":eligible[-1]["elapsed_seconds"]/cap,
            "metrics":metrics,"primary_error":metrics["normalized_clean_mse"]}
        artifacts["selected_states"][key]=state
        artifacts["selected_predictions"][key]={**copy.deepcopy(predictions[str(chosen["step"])]),"grid":prediction}
    primary=summary["selections"]["2.0"]
    for field in ("selected_step","best_validation_mse","metrics","primary_error"):
        summary[field]=copy.deepcopy(primary[field])
    return cell,summary,artifacts,arrays


@pytest.mark.parametrize("model_id",["fixed16","vp_centers","vp_widths","vp_both","joint_both","fixed32","fixed48","fourier16","fourier32","fourier48"])
def test_independent_reconstruction_all_models(config,model_id):
    cell,summary,artifacts,arrays=_fixture(config,model_id)
    result=audit.audit_fit(cell,summary,artifacts,arrays,config)
    assert result["max_prediction_discrepancy"]<1e-12
    assert result["max_ridge_weight_discrepancy"]<1e-12
    assert result["primary_error"]==summary["primary_error"]
    assert len(result["conditioning_by_checkpoint"])==len(summary["history"])
    if model_id=="joint_both":
        assert result["conditioning_by_checkpoint"][-1]["readout_stationarity_l2"]>1e-3


@pytest.mark.parametrize("corruption",["weight","prediction","geometry_leaf","dtype","metric","selection","eligibility","time_start","late_final","missing_checkpoint","missing_update","terminal_state","initial_state","condition","update_objective"])
def test_corrupted_fit_is_rejected(config,corruption):
    cell,summary,artifacts,arrays=_fixture(config)
    if corruption=="weight":artifacts["checkpoint_states"]["10"]["readout_weight"][0]+=1e-3
    elif corruption=="prediction":artifacts["checkpoint_predictions"]["20"]["validation"][0]+=1e-3
    elif corruption=="geometry_leaf":artifacts["checkpoint_states"]["10"]["basis_state"]["bank.centers"][0]+=1e-3
    elif corruption=="dtype":artifacts["checkpoint_states"]["10"]["readout_weight"]=artifacts["checkpoint_states"]["10"]["readout_weight"].float()
    elif corruption=="metric":summary["selections"]["2.0"]["primary_error"]+=.001
    elif corruption=="selection":summary["selections"]["2.0"]["selected_step"]=20
    elif corruption=="eligibility":summary["history"][-1]["admissible_time_budgets"]=[2.]
    elif corruption=="time_start":summary["history"][-1]["checkpoint_start_elapsed_seconds"]=.1
    elif corruption=="late_final":summary["history"][-1]["elapsed_seconds"]=.49
    elif corruption=="missing_checkpoint":del artifacts["checkpoint_states"]["10"]
    elif corruption=="missing_update":summary["update_history"].pop(3)
    elif corruption=="terminal_state":artifacts["final_state"]["readout_bias"]+=.1
    elif corruption=="initial_state":artifacts["initial_state"]["readout_bias"]+=.1
    elif corruption=="condition":summary["history"][0]["readout_diagnostics"]["normal_matrix_condition_2"]*=2
    elif corruption=="update_objective":summary["update_history"][0]["objective"]+=.01
    with pytest.raises(ValueError):audit.audit_fit(cell,summary,artifacts,arrays,config)


def test_ties_earliest_and_late_better_never_selected():
    caps=[.25,.75,2.]
    history=[{"step":step,"checkpoint_start_elapsed_seconds":time-.01,"elapsed_seconds":time,
        "validation_mse":score,"admissible_time_budgets":[cap for cap in caps if time<=cap]}
        for step,time,score in ((0,.1,1.),(10,.6,1.),(20,2.01,.001))]
    selected=audit.replay_selections(history,caps)
    assert {item["step"] for item in selected.values()}=={0}
    caps=[.01,.25,.75,2.]
    for row in history:row["admissible_time_budgets"]=[cap for cap in caps if row["elapsed_seconds"]<=cap]
    assert audit.replay_selections(history,caps)["0.01"] is None


@pytest.mark.parametrize("model_id,field",[("vp_centers","kappas"),("vp_widths","centers")])
def test_frozen_ablation_coordinate_cannot_drift(config,model_id,field):
    cell,summary,artifacts,arrays=_fixture(config,model_id)
    state=artifacts["checkpoint_states"]["10"]
    state[field][0]+=.01
    leaf="bank.centers" if field=="centers" else "bank.fixed_kappa"
    state["basis_state"][leaf].reshape(-1)[0]+=.01
    with pytest.raises(ValueError,match="Frozen geometry changed"):
        audit.audit_fit(cell,summary,artifacts,arrays,config)


def _balanced_rows(config,phase="development"):
    rows=[]
    for spec in config["models"]:
        for f,family in enumerate(config["tasks"]["families"]):
            for rep in range(config["tasks"]["replicates"][phase]):
                for noise in config["tasks"]["noise_levels"]:
                    for labels in config["tasks"]["label_budgets"]:
                        for init in range(config["tasks"]["initializations"][phase]):
                            error=1.+f+.1*rep+noise+.0001*labels+.01*init
                            factor={"vp_centers":.75,"vp_widths":.75,"vp_both":.7,"fixed32":.9,"fixed48":.95,"fourier32":.9,"fourier48":.85}.get(spec["id"],1.)
                            rows.append({"audited":True,"cell":{"phase":phase,"model_id":spec["id"],"family":family,"replicate":rep,
                                "noise_std":noise,"total_labels":labels,"initialization":init},
                                "timepoints":{str(float(cap)):{"available":True,"primary_error":factor*error} for cap in config["training"]["time_budgets_seconds"]}})
    return rows


def test_balanced_bundles_selection_and_bootstrap(config):
    config["tasks"]["label_budgets"]=[64,128,512]
    rows=_balanced_rows(config)
    result=audit.aggregate(rows,config,2.)
    models=result["models"]
    expected=2.+.1+.125+np.mean([64,128,512])*.0001+.005
    assert models["fixed16"]["balanced_periodic_error"]==pytest.approx(expected)
    assert models["fixed16"]["task_errors"]["smooth"]["1"]==pytest.approx(expected-1)
    choices=audit.choose_development(models,config)
    assert choices=={"vp_model":"vp_both","capacity_comparator":"fourier48"}
    result=audit.contrast(models,"vp_both","fixed16",config)
    assert result["relative_improvement"]==pytest.approx(.3)
    assert result["bootstrap"]["lower"]==pytest.approx(.3)
    assert result["bootstrap"]["upper"]==pytest.approx(.3)
    assert result["bootstrap"]["resamples"]==10000
    assert result["gate_passed"]
    assert result==audit.contrast(models,"vp_both","fixed16",config)


def test_comparator_ties_follow_config_order_and_counts(config):
    models={spec["id"]:{"balanced_periodic_error":1.} for spec in config["models"]}
    assert audit.choose_development(models,config)=={"vp_model":"vp_centers","capacity_comparator":"fixed32"}
    models["vp_widths"]["balanced_periodic_error"]=.5
    models["fourier48"]["balanced_periodic_error"]=.01
    assert audit.choose_development(models,config)=={"vp_model":"vp_widths","capacity_comparator":"fixed32"}


def test_missing_prefix_does_not_average_favorable_survivors(config):
    rows=_balanced_rows(config)
    victim=next(row for row in rows if row["cell"]["model_id"]=="fixed16")
    victim["timepoints"]["0.25"]={"available":False,"primary_error":None}
    result=audit.aggregate(rows,config,.25)
    assert result["complete"] is False
    assert result["models"]["fixed16"]["balanced_periodic_error"] is None
    assert result["models"]["fixed16"]["task_errors"]["smooth"].get("0") is None
    assert any(cell["mean_error"] is None for cell in result["cells"])


def test_empty_truncated_duplicate_and_pilot_aggregate_are_not_complete(config):
    rows=_balanced_rows(config)
    assert audit.aggregate(rows,config,2.)["complete"]
    for missing in ([],rows[:-1],rows+[rows[0]]):
        assert not audit.aggregate(missing,config,2.)["complete"]
    with pytest.raises(ValueError,match="Pilot roles"):
        audit.aggregate(_balanced_rows(config,"pilot"),config,2.)


def test_bootstrap_pairs_targets_and_rejects_unmatched(config):
    models=audit.aggregate(_balanced_rows(config),config,2.)["models"]
    del models["fixed16"]["task_errors"]["smooth"]["0"]
    with pytest.raises(ValueError,match="Unpaired"):audit.contrast(models,"vp_both","fixed16",config)


def test_unchanged_research_gate_can_fail_clean_audit(config):
    models=audit.aggregate(_balanced_rows(config),config,2.)["models"]
    result=audit.contrast(models,"fixed16","vp_both",config)
    assert result["relative_improvement"]<0
    assert not result["gate_passed"]


def _saved_pools(config,cell):
    """Manufacture only observations and deterministic analytic fixture targets."""
    roots=config["seeds"][cell["phase"]];phase_id={"pilot":1,"development":2,"confirmation":3}[cell["phase"]]
    streams={name:[roots["task" if index==0 else "observation"],2,phase_id,1,cell["replicate"],index]
        for name,index in {"function":0,"train_inputs":1,"train_noise":2,"validation_inputs":3,"validation_noise":4,"grid_noise":5}.items()}
    pools={};largest=max(config["tasks"]["label_budgets"])
    for split,count in (("train",3*largest//4),("validation",largest//4),("grid",config["tasks"]["grid_points"])):
        if split=="grid":x=(-math.pi+(np.arange(count)+.5)*2*math.pi/count)[:,None]
        else:x=np.random.default_rng(np.random.SeedSequence(streams[split+"_inputs"])).uniform(-math.pi,math.pi,(count,1))
        pools[split+"_x"]=x
        pools[split+"_clean"]=np.sin(x)+.2*np.cos(3*x)
        pools[split+"_unit_noise"]=np.random.default_rng(np.random.SeedSequence(streams[split+"_noise"])).standard_normal((count,1))
    function={"kind":"manufactured_fixture_no_registered_target_generation"}
    value={"dtype":"float64","label_budgets":config["tasks"]["label_budgets"],"noise_levels":config["tasks"]["noise_levels"],
        "function":function,"function_sha256":audit.object_hash(function),
        "arrays":{name:array.tolist() for name,array in pools.items()},
        "array_sha256":{name:hashlib.sha256(array.tobytes()).hexdigest() for name,array in pools.items()},
        "metadata":{"config_sha256":audit.object_hash(config),"phase":cell["phase"],"family":cell["family"],
            "replicate":cell["replicate"],"confirmation_lock_sha256":None,"stream_components":streams,
            "task_root":roots["task"],"observation_root":roots["observation"]}}
    value["sha256"]=audit.object_hash(value)
    return value


def test_saved_arrays_streams_hashes_nesting_and_noise(config):
    config["tasks"]["label_budgets"]=[64,128,512]
    cell=_fixture(config)[0];saved=_saved_pools(config,cell)
    first=audit._task_arrays(saved,config,cell,None)
    second=audit._task_arrays(saved,config,{**cell,"total_labels":512,"noise_std":.2},None)
    np.testing.assert_array_equal(first["train_x"],second["train_x"][:48])
    clean=np.array(saved["arrays"]["train_clean"][:48])
    np.testing.assert_allclose((first["train_y"]-clean)*4,second["train_y"][:48]-clean,atol=1e-15)
    saved["arrays"]["train_x"][0][0]+=.01
    array=np.array(saved["arrays"]["train_x"])
    saved["array_sha256"]["train_x"]=hashlib.sha256(array.tobytes()).hexdigest()
    saved["sha256"]=audit.object_hash({key:value for key,value in saved.items() if key!="sha256"})
    with pytest.raises(ValueError,match="Independent input"):audit._task_arrays(saved,config,cell,None)


@pytest.fixture
def invocation(config,tmp_path,monkeypatch):
    config["models"]=[config["models"][0]]
    config["tasks"].update(families=["smooth"],noise_levels=[.05],pilot_label_budgets=[64],pilot_timing_label_budget=64)
    cells=audit.planned_runs(config,"pilot")
    run=tmp_path/"run";run.mkdir();(run/"sources").mkdir()
    (tmp_path/"source.py").write_text("fixture_source=1\n")
    (run/"sources"/"source.py").write_text("fixture_source=1\n")
    sources={"source.py":audit.file_hash(tmp_path/"source.py")}
    monkeypatch.setattr(audit,"ROOT",tmp_path)
    monkeypatch.setattr(audit,"SOURCE_PATHS",("source.py",))
    monkeypatch.setattr(audit,"source_manifest",lambda:sources)
    monkeypatch.setattr(audit,"checked_path",lambda path:tmp_path/path)
    monkeypatch.setattr(audit,"verify_dependencies",lambda values:[None if audit.file_hash(tmp_path/path)==digest else (_ for _ in ()).throw(ValueError("Dependency changed")) for path,digest in values.items()])
    # Unit fixture isolates the numerical observer; protocol verifier has its own
    # real source/receipt/deadline corruption tests. Include its receipt addition.
    def verify(path,*_):
        value=audit.read_json(path)
        value["evidence_hashes"]["run/completion.json"]=audit.file_hash(run/"completion.json")
        return value,[]
    monkeypatch.setattr(audit,"verify_run_manifest",verify)
    saved=_saved_pools(config,cells[0]);audit.write_json(run/"task.json",saved)
    evidence={"run/task.json":audit.file_hash(run/"task.json")};rows=[]
    for cell in cells:
        arrays=audit._task_arrays(saved,config,cell,None)
        _,summary,artifacts,_=_fixture(config,"fixed16",arrays=arrays,cell=cell)
        folder=run/"fits"/cell["id"];folder.mkdir(parents=True)
        audit.write_json(folder/"summary.json",summary);torch.save(artifacts,folder/"artifacts.pt")
        paths={"task_path":"run/task.json","summary_path":(folder/"summary.json").relative_to(tmp_path).as_posix(),
            "artifact_path":(folder/"artifacts.pt").relative_to(tmp_path).as_posix()}
        row={"cell":cell,"status":"completed",**paths}
        for name in ("summary","artifact"):
            row[name+"_hash"]=audit.file_hash(tmp_path/paths[name+"_path"]);evidence[paths[name+"_path"]]=row[name+"_hash"]
        rows.append(row)
    manifest={"kind":"locality_run_v2","phase":"pilot","config_hash":audit.object_hash(config),
        "source_hashes":sources,"evidence_hashes":evidence,"prerequisite_hashes":{},"lock_hash":None,
        "learning_rates":None,"rows":rows,"planned_count":len(cells),"complete":True,"status":"completed","deadline_respected":True}
    audit.write_json(run/"manifest.json",manifest);audit.write_json(run/"config.json",config)
    audit.write_json(run/"completion.json",{"fixture":True})
    return run/"manifest.json",config


def test_complete_manufactured_manifest_binds_receipt_and_does_not_select_pilots(invocation):
    path,config=invocation
    result=audit.analyze_run(path,config)
    assert result["complete"],result["audit_errors"]
    assert result["audit"]["all_passed"] is True
    assert result["choices"] is None
    assert not result["methodological_gate_evaluated"]
    assert result["input_manifest"]=="run/manifest.json"
    assert "run/completion.json" in result["evidence_hashes"]
    assert result["learning_rates"] is None
    audit.object_hash(result)


@pytest.mark.parametrize("corruption",["missing_row","duplicate_row","source","artifact","failed_row","summary"])
def test_manifest_corruption_retains_failure(invocation,corruption):
    path,config=invocation;manifest=audit.read_json(path)
    if corruption=="missing_row":manifest["rows"].pop()
    elif corruption=="duplicate_row":manifest["rows"].append(copy.deepcopy(manifest["rows"][0]))
    elif corruption=="source":(path.parent/"sources"/"source.py").write_text("changed")
    elif corruption=="artifact":(audit.ROOT/manifest["rows"][0]["artifact_path"]).write_bytes(b"changed")
    elif corruption=="failed_row":manifest["rows"][0]["status"]="failed"
    elif corruption=="summary":(audit.ROOT/manifest["rows"][0]["summary_path"]).write_text("{}")
    path.write_text(__import__("json").dumps(manifest))
    result=audit.analyze_run(path,config)
    assert not result["complete"] and not result["audit"]["all_passed"]
    assert result["audit_errors"]
    assert len(result["rows"])==len(manifest["rows"])


def test_confirmation_refuses_missing_lock_before_artifact_reads(invocation):
    path,config=invocation;manifest=audit.read_json(path);manifest["phase"]="confirmation"
    path.write_text(__import__("json").dumps(manifest))
    with pytest.raises(ValueError,match="exact invocation lock"):audit.analyze_run(path,config)
