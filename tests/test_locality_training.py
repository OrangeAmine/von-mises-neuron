"""CPU-only disposable-fixture checks of the timed locality trainer."""

from __future__ import annotations

import copy
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import src.locality_training as training
from src.locality_models import circular_features, solve_readout


@pytest.fixture
def config():
    value = json.loads((Path(__file__).resolve().parents[1]/'configs/vm_locality_v2.json').read_text())
    value['training'].update(max_updates=3,validation_every=1,time_budgets_seconds=[100.],primary_time_seconds=100.)
    return value


@pytest.fixture
def arrays():
    train_x = np.linspace(-2.7,2.8,11)[:,None]
    validation_x = np.linspace(-2.4,2.5,7)[:,None]
    grid_x = np.linspace(-math.pi,math.pi,31,endpoint=False)[:,None]
    target = lambda x: np.sin(x)+.3*np.cos(3*x)
    return {'train_x':train_x,'train_y':target(train_x)+.03*np.arange(11)[:,None]/11,
            'validation_x':validation_x,'validation_y':target(validation_x),
            'grid_x':grid_x,'grid_clean':target(grid_x),'grid_noisy':target(grid_x)+.1}


@pytest.fixture
def initialization():
    centers = (-math.pi+np.arange(16)*2*math.pi/16+.19)[:,None]
    value = {'centers':centers,'kappas':np.full((16,1),4.)}
    for count, widths in [(32,[4.,16.]),(48,[1.,4.,16.])]:
        value['fixed'+str(count)] = {'centers':np.repeat(centers,len(widths),axis=0),
                                    'kappas':np.tile(widths,16)[:,None]}
    return value


class TickClock:
    def __init__(self, increment=.1):
        self.value = -increment
        self.increment = increment

    def __call__(self):
        self.value += self.increment
        return self.value


def fit(model_id, arrays, initialization, config, **kwargs):
    return training.fit_locality(model_id,arrays,initialization,config,
                                 learning_rate=.03,device='cpu',**kwargs)


@pytest.mark.parametrize('model_id,basis,fitted,optimized',[
    ('fixed16',16,17,0),('vp_centers',16,33,16),('vp_widths',16,33,16),
    ('vp_both',16,49,32),('joint_both',16,49,49),('fixed32',32,33,0),
    ('fixed48',48,49,0),('fourier16',16,17,0),('fourier32',32,33,0),('fourier48',48,49,0)])
def test_every_model_fits_and_replays(model_id,basis,fitted,optimized,arrays,initialization,config):
    bundle = fit(model_id,arrays,initialization,config)
    summary, artifacts = bundle['summary'],bundle['artifacts']
    assert summary['status']=='completed',summary
    assert (summary['basis_count'],summary['fitted_parameter_count'],summary['optimized_parameter_count'])==(basis,fitted,optimized)
    assert summary['ridge_alpha']==pytest.approx(len(arrays['train_x'])*.001)
    assert len(summary['update_history'])==(3 if optimized else 0)
    assert len(summary['history'])==(4 if optimized else 1)
    assert summary['primary_error']>=0
    for row in summary['history']:
        state = artifacts['checkpoint_states'][str(row['step'])]
        for split in ('train','validation'):
            predictions = training.predict_locality_state(state,arrays[split+'_x'])
            torch.testing.assert_close(predictions,artifacts['checkpoint_predictions'][str(row['step'])][split],rtol=1e-12,atol=1e-12)
            mse = np.mean((predictions.numpy()-arrays[split+'_y'])**2)
            assert row[split+'_mse']==pytest.approx(mse,abs=1e-12)
        assert state['readout_matches_geometry']
    json.dumps(summary,allow_nan=False)
    stream = io.BytesIO()
    torch.save(artifacts,stream)
    stream.seek(0)
    recovered = torch.load(stream,weights_only=True)
    assert set(recovered['checkpoint_states'])==set(artifacts['checkpoint_states'])


def test_initial_functions_and_ridge_solutions_identical(arrays,initialization,config):
    config['training']['max_updates']=0
    bundles = [fit(model,arrays,initialization,config) for model in ['fixed16','vp_centers','vp_widths','vp_both','joint_both']]
    reference = bundles[0]['artifacts']['initial_state']
    for bundle in bundles[1:]:
        state = bundle['artifacts']['initial_state']
        torch.testing.assert_close(state['readout_weight'],reference['readout_weight'],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(state['readout_bias'],reference['readout_bias'],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(training.predict_locality_state(state,arrays['grid_x']),
                                   training.predict_locality_state(reference,arrays['grid_x']),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('model,frozen,learned',[('vp_centers','kappas','centers'),('vp_widths','centers','kappas')])
def test_only_requested_geometry_changes(model,frozen,learned,arrays,initialization,config):
    bundle=fit(model,arrays,initialization,config)
    initial,final=bundle['artifacts']['initial_state'],bundle['artifacts']['final_state']
    assert torch.equal(initial[frozen],final[frozen])
    assert not torch.equal(initial[learned],final[learned])
    names=[row['name'] for row in bundle['summary']['optimizer_parameters']]
    assert all(name.startswith('basis.') for name in names)
    assert len(names)==1


def test_joint_never_refits_after_initialization(monkeypatch,arrays,initialization,config):
    original=training.solve_readout
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'solve_readout',counted)
    result=fit('joint_both',arrays,initialization,config)
    assert result['summary']['status']=='completed'
    assert len(calls)==1
    state=result['artifacts']['final_state']
    features=circular_features(torch.from_numpy(arrays['train_x']),state['centers'],state['kappas'])
    refit_weight,_=solve_readout(features,torch.from_numpy(arrays['train_y']),.001)
    assert torch.linalg.vector_norm(refit_weight-state['readout_weight'])>1e-6


def test_vp_readout_fresh_at_every_checkpoint(monkeypatch,arrays,initialization,config):
    original=training.solve_readout
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'solve_readout',counted)
    result=fit('vp_both',arrays,initialization,config)
    assert result['summary']['status']=='completed'
    assert len(calls)==1+2*config['training']['max_updates']
    assert all(row['readout_diagnostics']['readout_stationarity_l2']<1e-12 for row in result['summary']['history'])


def test_joint_and_vp_initial_geometry_update_equal(arrays,initialization,config):
    config['training']['max_updates']=1
    a,b=[fit(model,arrays,initialization,config) for model in ['vp_both','joint_both']]
    for key in ('centers','kappas'):
        torch.testing.assert_close(a['artifacts']['final_state'][key],b['artifacts']['final_state'][key],atol=1e-12,rtol=1e-12)


def controlled_validation(monkeypatch,arrays,scores):
    original=training.predict_readout
    counter=iter(scores)
    arrays['validation_y']=np.zeros_like(arrays['validation_y'])
    def overridden(features,weight,bias):
        if len(features)==len(arrays['validation_x']):
            return torch.full((len(features),1),math.sqrt(next(counter)),dtype=torch.float64)
        return original(features,weight,bias)
    monkeypatch.setattr(training,'predict_readout',overridden)


def test_late_better_checkpoint_never_admitted(monkeypatch,arrays,initialization,config):
    # Calls: start0, checkpoint0 start/end .1/.2, update1 .3,
    # checkpoint1 .4/.5, update2 .6, checkpoint2 .7/.8 (late).
    config['training'].update(time_budgets_seconds=[.25,.55,.75],primary_time_seconds=.75)
    controlled_validation(monkeypatch,arrays,[3.,2.,1.])
    result=fit('vp_both',arrays,initialization,config,clock=TickClock())
    summary=result['summary']
    assert summary['status']=='completed'
    assert summary['stop_reason']=='wall_time'
    assert summary['updates_completed']==2
    assert summary['selected_step']==1
    assert summary['selections']['0.25']['selected_step']==0
    assert summary['selections']['0.55']['selected_step']==1
    assert summary['selections']['0.75']['selected_step']==1
    assert summary['overshoot_seconds']==pytest.approx(.05)
    assert summary['history'][-1]['admissible_time_budgets']==[]
    assert '2' in result['artifacts']['checkpoint_states']
    assert result['artifacts']['final_state'] is result['artifacts']['checkpoint_states']['2']


def test_exact_validation_ties_keep_earliest(monkeypatch,arrays,initialization,config):
    controlled_validation(monkeypatch,arrays,[1.,1.,1.,1.])
    result=fit('vp_centers',arrays,initialization,config,clock=TickClock())
    assert result['summary']['selected_step']==0
    assert result['summary']['selections']['100.0']['eligible_checkpoints']==4


def test_small_budget_unavailable_does_not_fail_primary(arrays,initialization,config):
    config['training'].update(time_budgets_seconds=[.05,1.],primary_time_seconds=1.)
    result=fit('fixed16',arrays,initialization,config,clock=TickClock())
    assert result['summary']['status']=='completed'
    assert result['summary']['selections']['0.05']['status']=='budget_unavailable'
    assert result['summary']['selections']['1.0']['status']=='completed'
    assert '0.05' not in result['artifacts']['selected_states']
    assert result['summary']['stop_reason']=='fixed_closed_form'
    assert result['summary']['fit_seconds']==pytest.approx(.2)


@pytest.mark.parametrize('model',['fixed16','vp_both'])
def test_no_admissible_initialization_is_explicit_unavailable(model,arrays,initialization,config):
    config['training'].update(time_budgets_seconds=[.05],primary_time_seconds=.05)
    result=fit(model,arrays,initialization,config,clock=TickClock())
    assert result['summary']['status']=='budget_unavailable'
    assert result['summary']['updates_completed']==0
    assert result['summary']['selected_step'] is None
    assert 'primary_error' not in result['summary']
    assert set(result['artifacts']['checkpoint_states'])=={'0'}


def test_atomic_block_retains_all_updates_until_evaluation(arrays,initialization,config):
    config['training'].update(max_updates=5,validation_every=3,time_budgets_seconds=[.25],primary_time_seconds=.25)
    result=fit('vp_widths',arrays,initialization,config,clock=TickClock())
    assert result['summary']['updates_completed']==3
    assert [row['step'] for row in result['summary']['history']]==[0,3]
    assert result['summary']['history'][-1]['elapsed_seconds']==pytest.approx(.7)
    assert result['summary']['selected_step']==0


def test_final_nonmultiple_update_checkpoint_preserved(arrays,initialization,config):
    config['training'].update(max_updates=5,validation_every=3)
    result=fit('joint_both',arrays,initialization,config,clock=TickClock())
    assert [row['step'] for row in result['summary']['history']]==[0,3,5]
    assert result['summary']['updates_completed']==5


def test_clean_target_changes_cannot_change_selection(arrays,initialization,config):
    changed=copy.deepcopy(arrays)
    changed['grid_clean']=7*changed['grid_clean']+21
    changed['grid_noisy']=-4*changed['grid_noisy']-8
    a,b=[fit('vp_both',values,initialization,config,clock=TickClock()) for values in (arrays,changed)]
    assert a['summary']['history']==b['summary']['history']
    assert a['summary']['update_history']==b['summary']['update_history']
    assert a['summary']['selected_step']==b['summary']['selected_step']
    for name in ('centers','kappas','readout_weight','readout_bias'):
        assert torch.equal(a['artifacts']['final_state'][name],b['artifacts']['final_state'][name])
    assert a['summary']['primary_error']!=b['summary']['primary_error']


@pytest.mark.parametrize('error,expected',[(FloatingPointError('nonfinite fixture'),'failed'),(KeyboardInterrupt('fixture interrupt'),'interrupted')])
def test_failed_update_preserves_evidence_and_cannot_succeed(monkeypatch,error,expected,arrays,initialization,config):
    original=training._update
    calls=[]
    def failing(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            raise error
        return original(*args,**kwargs)
    monkeypatch.setattr(training,'_update',failing)
    result=fit('vp_centers',arrays,initialization,config,clock=TickClock())
    assert result['summary']['status']==expected
    assert result['summary']['failed_step']==2
    assert result['summary']['updates_completed']==1
    assert len(result['summary']['history'])==2
    assert 'initial_state' in result['artifacts'] and 'final_state' in result['artifacts']
    assert 'primary_error' not in result['summary']
    assert len(calls)==2
    json.dumps(result['summary'],allow_nan=False)


def test_input_arrays_and_initialization_are_not_modified(arrays,initialization,config):
    original=copy.deepcopy((arrays,initialization))
    result=fit('joint_both',arrays,initialization,config)
    assert result['summary']['status']=='completed'
    for name in arrays:
        np.testing.assert_array_equal(arrays[name],original[0][name])
    np.testing.assert_array_equal(initialization['centers'],original[1]['centers'])
    np.testing.assert_array_equal(initialization['kappas'],original[1]['kappas'])


def test_nonfinite_diagnostics_never_enter_json_summary(monkeypatch,arrays,initialization,config):
    original=training.readout_diagnostics
    def nonfinite(*args,**kwargs):
        value=original(*args,**kwargs)
        value['normal_matrix_condition_2']=float('inf')
        return value
    monkeypatch.setattr(training,'readout_diagnostics',nonfinite)
    result=fit('fixed16',arrays,initialization,config)
    assert result['summary']['status']=='failed'
    assert 'Nonfinite checkpoint diagnostics' in result['summary']['failure']
    assert not result['summary']['history']
    assert 'final_state' in result['artifacts']
    json.dumps(result['summary'],allow_nan=False)


def test_cpu_warmup_has_no_rng_effect():
    before=torch.random.get_rng_state().clone()
    result=training.warmup_locality('cpu')
    assert torch.equal(before,torch.random.get_rng_state())
    assert result['seconds']>0
    assert result['device']=='cpu'


@pytest.mark.parametrize('field,value',[('native_weight_decay',.001),('bias_penalized',True),
    ('dtype','float32'),('early_stopping',True),('optimizer','SGD'),('penalty',0),('clip_gradient_norm',float('nan'))])
def test_invalid_scientific_configuration_retains_failed_record(field,value,arrays,initialization,config):
    config['training'][field]=value
    result=fit('vp_both',arrays,initialization,config)
    assert result['summary']['status']=='failed'
    assert result['summary']['updates_completed']==0
    assert not result['artifacts']['checkpoint_states']
    json.dumps(result['summary'],allow_nan=False)
