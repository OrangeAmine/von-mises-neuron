"""Independent auditor checks with tiny CPU fixtures and disposable pilot tasks."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from scripts.analyze_vm_methodology import (
    _audit_linear_optimum, _audit_task, _expected_cells, _identity,
    analyze_a, analyze_b, audit_fit, balanced_b, bootstrap_improvement,
    reconstruct_prediction, replay_pilot_rates, replay_validation_selection, select_ridge_lambda,
)
from src.methodology_protocol import planned_runs
from src.methodology_tasks import batch_seed, generate_task, initialization_bundle
from src.methodology_training import closed_form_a, fit_ridge, make_a_model, make_b_model, train_a, train_b


@pytest.fixture(scope="module",autouse=True)
def cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/vm_methodology_v1.json").read_text())


def test_validation_lowest_checkpoint_and_separate_patience_anchor():
    history=[{"step":0,"validation_mse":1.,"selected":True,"significant_improvement":True},
             {"step":20,"validation_mse":1.-.2e-6,"selected":True,"significant_improvement":False},
             {"step":40,"validation_mse":1.-.4e-6,"selected":True,"significant_improvement":False},
             {"step":60,"validation_mse":1.-1.2e-6,"selected":True,"significant_improvement":True}]
    selected,best,stale=replay_validation_selection(history,1e-6,5)
    assert selected==60 and best==1.-1.2e-6 and stale==0
    assert replay_validation_selection(history[:3],1e-6,5)[2]==2
    bad=copy.deepcopy(history)
    bad[1]["selected"]=False
    with pytest.raises(ValueError,match="selection"):
        replay_validation_selection(bad,1e-6,5)


def test_patience_cannot_continue_and_ties_keep_earliest():
    history=[{"step":i*20,"validation_mse":1.,"selected":i==0,"significant_improvement":i==0} for i in range(4)]
    assert replay_validation_selection(history[:3],1e-6,2)==(0,1.,2)
    with pytest.raises(ValueError,match="continued"):
        replay_validation_selection(history,1e-6,2)
    with pytest.raises(ValueError,match="initialization"):
        replay_validation_selection(history[1:],1e-6,2)


def test_ridge_ties_use_global_minimum():
    history=[{"lambda":0.,"validation_mse":1.+1.5e-12},
             {"lambda":.0001,"validation_mse":1.+.8e-12},
             {"lambda":.01,"validation_mse":1.},
             {"lambda":1.,"validation_mse":2.}]
    assert select_ridge_lambda(history)==.0001
    with pytest.raises(ValueError):
        select_ridge_lambda([{"lambda":0,"validation_mse":math.nan}])


@pytest.mark.parametrize("stage,phase",[(stage,phase) for stage in ("A","B") for phase in ("pilot","development","confirmation")])
def test_independent_matrix_enumeration_matches_canonical_counts(config,stage,phase):
    choices={"A":{"Adam":.01,"SGD":.03},"B":{name:.01 for name in ("learned_vm","learned_geodesic","sincos_mlp","raw_angle_mlp")}}
    actual=planned_runs(config,stage,phase,choices)
    expected=_expected_cells(config,stage,phase,choices)
    assert sorted(map(repr,expected))==sorted(map(repr,(_identity(row) for row in actual)))
    assert len(set(expected))==len(expected)
    # Enumerating the registry is allowed; this test generates no held-out function.


@pytest.mark.parametrize("model",["polar","cartesian"])
def test_independent_a_curve_and_optimum_reconstruction(config,model):
    tiny=copy.deepcopy(config)
    tiny["A"]["pilot"]["updates"]=2
    tiny["A"]["record_every"]=1
    task=generate_task(tiny,"A","pilot","phase_only")
    p=task.function.parameters
    network=make_a_model(model,p["amplitude_before"],p["phase_before"],device="cpu")
    bundle=train_a(network,task,optimizer_name="Adam",learning_rate=.01,penalty=0.,updates=2,record_every=1,device="cpu")
    row={"stage":"A","phase":"pilot","family":"phase_only","replicate":0,"model":model,"kind":"gradient",
         "optimizer":"Adam","learning_rate":.01,"penalty":0.,"summary":bundle["summary"]}
    audit=audit_fit(row,bundle["artifacts"],task.as_arrays(),task.to_manifest(),tiny)
    assert audit["primary_error"]==pytest.approx(bundle["summary"]["primary_error"])
    broken=copy.deepcopy(bundle["summary"])
    broken["history"][1]["clean_mse"]+=.01
    with pytest.raises(ValueError,match="A curve"):
        audit_fit(dict(row,summary=broken),bundle["artifacts"],task.as_arrays(),task.to_manifest(),tiny)
    ref=closed_form_a(task,penalty=.001)
    ref_row={"stage":"A","phase":"pilot","family":"phase_only","replicate":0,"model":"reference","kind":"reference","penalty":.001,"summary":ref["summary"]}
    assert audit_fit(ref_row,ref["artifacts"],task.as_arrays(),task.to_manifest(),tiny)["selection_replayed"]


@pytest.mark.parametrize("model",["learned_vm","learned_geodesic","sincos_mlp","raw_angle_mlp"])
def test_independent_b_state_selection_and_batch_reconstruction(config,model):
    tiny=copy.deepcopy(config)
    tiny["B"]["pilot"]["max_updates"]=2
    tiny["B"]["optimizer"]["validation_every"]=1
    task=generate_task(tiny,"B","pilot","narrow")
    start=initialization_bundle(tiny,task)
    network=make_b_model(model,start,device="cpu")
    bundle=train_b(network,task,learning_rate=.01,batch_seed=batch_seed(tiny,task),max_updates=2,validation_every=1,device="cpu")
    row={"stage":"B","phase":"pilot","family":"narrow","replicate":0,"model":model,"kind":"gradient",
         "learning_rate":.01,"n_train":128,"initialization_index":0,"summary":bundle["summary"]}
    assert bundle["summary"]["status"]=="completed"
    audit=audit_fit(row,bundle["artifacts"],task.as_arrays(),task.to_manifest(),tiny)
    assert audit["primary_error"]==pytest.approx(bundle["summary"]["primary_error"])
    artifacts=copy.deepcopy(bundle["artifacts"])
    artifacts["batch_indices"][0,0]=(artifacts["batch_indices"][0,0]+1)%128
    with pytest.raises(ValueError,match="cyclic batches"):
        audit_fit(row,artifacts,task.as_arrays(),task.to_manifest(),tiny)


@pytest.mark.parametrize("model",["fourier_ridge","fixed_vm_ridge"])
def test_independent_ridge_all_candidates_and_minimum_norm(config,model):
    task=generate_task(config,"B","pilot","multimodal").train_prefix(32)
    bundle=fit_ridge(model,task)
    row={"stage":"B","phase":"pilot","family":"multimodal","replicate":0,"model":model,"kind":"ridge","n_train":32,"summary":bundle["summary"]}
    assert bundle["summary"]["status"]=="completed"
    result=audit_fit(row,bundle["artifacts"],task.as_arrays(),task.to_manifest(),config)
    assert result["selection_replayed"]
    bad=copy.deepcopy(bundle["summary"])
    bad["history"][0]["alpha"]=3.
    with pytest.raises(ValueError,match="alpha"):
        audit_fit(dict(row,summary=bad),bundle["artifacts"],task.as_arrays(),task.to_manifest(),config)


@pytest.mark.parametrize("stage,family",[("A","phase_only"),("B","smooth"),("B","narrow"),("B","multimodal"),("B","nonperiodic")])
def test_independent_task_equations_noise_and_standardization(config,stage,family):
    task=generate_task(config,stage,"pilot",family)
    _audit_task(task.as_arrays(),task.to_manifest(),config,stage,"pilot",family,0)
    arrays={name:value.copy() for name,value in task.as_arrays().items()}
    arrays["train_y"][0]+=1
    with pytest.raises(ValueError,match="train response"):
        _audit_task(arrays,task.to_manifest(),config,stage,"pilot",family,0)


def b_rows(scale=1.):
    rows=[]
    for model,base in (("learned_vm",.6),("fourier_ridge",1.),("fixed_vm_ridge",1.2),("learned_geodesic",1.4),("sincos_mlp",1.6),("raw_angle_mlp",.01)):
        for family in ("smooth","narrow","multimodal","nonperiodic"):
            for task in range(2):
                for size in (32,128,512):
                    for initialization in range(1 if "ridge" in model else 2):
                        rows.append({"audited":True,"stage":"B","family":family,"replicate":task,"model":model,
                                     "n_train":size,"initialization_index":initialization,
                                     "primary_error":base*scale*(1+task),"noise_std":.05 if task==0 else .2})
    return rows


def test_balanced_aggregation_noise_subgroups_and_strong_comparator(config):
    rows=b_rows()
    result=analyze_b(rows,config,True)
    assert result["primary_comparator"]=="fourier_ridge"
    assert result["relative_improvement"]==pytest.approx(.4)
    assert result["gate_passed"]
    assert result["models"]["learned_vm"]["balanced_periodic_error"]==pytest.approx(.9)
    noise=result["models"]["learned_vm"]["noise_subgroups"]
    assert noise["0.05"]["balanced_periodic_error"]==pytest.approx(.6)
    assert noise["0.2"]["balanced_periodic_error"]==pytest.approx(1.2)
    # Raw-angle can beat VM without qualifying as the primary periodic comparator.
    assert result["comparisons"]["raw_angle_mlp"]<0
    assert not analyze_b(rows,config,False)["gate_passed"]


def test_b_gate_failure_is_not_replaced_by_other_control(config):
    result=analyze_b(b_rows(),config,True,comparator="fixed_vm_ridge")
    assert result["primary_comparator"]=="fixed_vm_ridge"
    rows=b_rows()
    for row in rows:
        if row["model"]=="learned_vm" and row["family"]=="smooth":
            row["primary_error"]*=2
    failed=analyze_b(rows,config,True)
    assert failed["relative_improvement"]>.1
    assert failed["family_improvements"]["smooth"]==pytest.approx(-.2)
    assert not failed["gate_passed"]


def test_task_bootstrap_preserves_pairing_and_is_reproducible():
    pairs={"f1":([.5,1.,2.],[1.,2.,4.]),"f2":([2.,3.,5.],[4.,6.,10.])}
    result=bootstrap_improvement(pairs,seed=9001,stage="B")
    assert result["resamples"]==10000
    assert result["lower"]==pytest.approx(.5) and result["upper"]==pytest.approx(.5)
    assert result==bootstrap_improvement(pairs,seed=9001,stage="B")
    undefined=bootstrap_improvement({"zero":([0.],[0.])},seed=9001,stage="A")
    assert undefined["lower"] is None


def test_a_gate_averages_whole_bundles_and_retains_specificity(config):
    rows=[]
    for condition in config["A"]["family_ids"]:
        for model,value in (("polar",.5),("cartesian",1.)):
            for replicate in range(2):
                rows.append({"audited":True,"stage":"A","kind":"gradient","family":condition,
                             "optimizer":"Adam","penalty":0.,"model":model,"replicate":replicate,
                             "primary_error":value*(replicate+1)})
    result=analyze_a(rows,config,True)
    assert result["gate_passed"] and result["relative_improvement"]==pytest.approx(.5)
    assert len(result["cells"])==8
    assert len(result["paired_task_differences"])==2
    assert not analyze_a(rows,config,False)["gate_passed"]
    for row in rows:
        if row["family"]=="both" and row["model"]=="polar":
            row["primary_error"]*=2.1
    assert not analyze_a(rows,config,True)["gate_passed"]


def test_pilot_incomplete_candidate_cannot_win(config):
    rows=[]
    for model in ("learned_vm","learned_geodesic","sincos_mlp","raw_angle_mlp"):
        for rate in (.001,.01):
            for family in config["B"]["family_ids"]:
                rows.append({"kind":"gradient","model":model,"learning_rate":rate,"audited":True,
                             "summary":{"best_validation_mse":.1 if rate==.01 else 1.}})
    rows[-1]["audited"]=False
    result=replay_pilot_rates(rows,config,"B")
    assert result["complete_selection"]
    assert result["choices"]["learned_vm"]==.01
    assert result["choices"]["raw_angle_mlp"]==.001


def test_stationarity_rejects_wrong_coefficients_and_unpenalized_bias():
    x=np.array([[-1.],[0.],[1.]])
    target=2*x+4
    coefficients=np.linalg.lstsq(np.column_stack((x,np.ones(3))),target,rcond=None)[0]
    _audit_linear_optimum(x,target,coefficients,0.)
    with pytest.raises(ValueError,match="stationarity"):
        _audit_linear_optimum(x,target,np.array([[2.],[0.]]),0.)


def test_ill_conditioned_backward_error_is_reported_without_forward_accuracy_claim():
    theta=np.linspace(-3.,3.,128)[:,None]
    features=np.column_stack((np.cos(theta),np.cos(theta)+1e-12*np.sin(theta)))
    target=np.sin(theta)
    design=np.column_stack((features,np.ones(len(features))))
    coefficients=np.linalg.lstsq(design,target,rcond=None)[0]
    diagnostics=_audit_linear_optimum(features,target,coefficients,0.)
    assert diagnostics["coefficient_max_absolute"]>1e11
    assert diagnostics["numerically_sensitive"]
    assert diagnostics["normwise_backward_error"]<diagnostics["float64_roundoff_bound"]
    assert diagnostics["coefficient_replay_max_difference"]==0
    assert "no coefficient/prediction forward-accuracy guarantee" in diagnostics["interpretation"]
    # Even perturbations hidden by condition-scaled stationarity cannot pass
    # exact replay of the recorded rounded-design solver.
    corrupted=coefficients.copy()
    corrupted[-1]+=.001
    with pytest.raises(ValueError,match="coefficient replay"):
        _audit_linear_optimum(features,target,corrupted,0.)


def test_normwise_stationarity_bound_rejects_large_finite_corruption():
    features=np.array([[-1.],[0.],[1.]])
    target=np.array([[2.],[3.],[4.]])
    with pytest.raises(ValueError,match="normwise stationarity"):
        _audit_linear_optimum(features,target,np.array([[50.],[100.]]),0.)
