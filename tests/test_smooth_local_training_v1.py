"""Budget selection, postfit separation and failure preservation for VM-N01."""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.smooth_local_training_v1 import budget_key, fit_smooth_local
from src.smooth_local_v1 import predict_smooth_local_state


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/vm_n01_v1.json").read_text())
SPECS = CONFIG["models"]


def arrays():
    train = np.linspace(-3, 3, 27)[:, None]
    validation = np.linspace(-2.95, 2.95, 17)[:, None]
    grid = np.linspace(-math.pi, math.pi, 101)[:, None]
    return {"train_x": train, "train_y": np.sin(train), "validation_x": validation,
            "validation_y": np.sin(validation), "grid_x": grid, "grid_clean": np.sin(grid)}


def settings(**updates):
    return {**CONFIG["training"], "time_budgets_seconds": [10., 20.],
            "max_updates": 4, "validation_every": 2, **updates}


class Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


@pytest.mark.parametrize("spec", SPECS, ids=lambda v: v["id"])
def test_all_registered_models_tiny_cpu_fit_replay_and_counts(spec):
    result = fit_smooth_local(spec, arrays(), .2, .01 if spec["kind"] != "ridge" else None, settings(), "cpu")
    assert result["status"] == "completed", result["failures"]
    assert result["basis_count"] == spec["features"]
    assert result["fitted_parameter_count"] == spec["parameters"]
    assert result["updates"] == (0 if spec["kind"] == "ridge" else 4)
    assert len(result["checkpoints"]) == (1 if spec["kind"] == "ridge" else 3)
    assert 0 <= result["fit_seconds"] <= result["total_seconds"]
    assert result["nonfit_seconds"] >= 0
    for key, index in result["selected"].items():
        row = result["checkpoints"][index]
        assert row["elapsed"] <= float(key)
        prediction = predict_smooth_local_state(row["state"], arrays()["grid_x"]).numpy()
        expected = np.mean((prediction-arrays()["grid_clean"])**2) / np.var(arrays()["grid_clean"])
        assert result["metrics"][key]["normalized_mse"] == pytest.approx(expected, rel=1e-12)


def test_late_checkpoints_are_preserved_and_ineligible_after_capture():
    spec = next(v for v in SPECS if v["id"] == "hybrid_joint")
    # Call order: start, checkpoint0, checkpoint2, checkpoint4, fit end, total end.
    clock = Clock([0., .1, .4, 1.1, 1.11, 1.12])
    result = fit_smooth_local(spec, arrays(), 0., .01,
        settings(time_budgets_seconds=[.25, 1.]), "cpu", clock=clock)
    assert result["status"] == "completed"
    assert [v["elapsed"] for v in result["checkpoints"]] == [.1, .4, 1.1]
    assert result["selected"]["0.25"] == 0
    assert result["selected"]["1"] in (0, 1)
    assert result["overshoot_seconds"] == pytest.approx(.11)
    assert result["max_block_seconds"] == pytest.approx(.7)


def test_no_initial_checkpoint_in_allowance_is_incomplete_not_success():
    result = fit_smooth_local(SPECS[0], arrays(), 0., None,
        settings(time_budgets_seconds=[.25, 1.]), "cpu", clock=Clock([0., 1.1, 1.11, 1.12]))
    assert result["status"] == "incomplete_budget"
    assert result["selected"] == {"0.25": None, "1": None}
    assert len(result["checkpoints"]) == 1 and not result["metrics"]


def test_equal_initial_joint_and_vp_ridge_states():
    results = [fit_smooth_local(next(v for v in SPECS if v["id"] == name), arrays(), .23, .01,
                               settings(max_updates=0), "cpu") for name in ("hybrid_joint", "hybrid_vp")]
    states = [result["checkpoints"][0]["state"]["state_dict"] for result in results]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


def test_joint_never_refits_readout_after_initialization(monkeypatch):
    import src.smooth_local_v1 as models
    calls = []
    original = models.SmoothLocalRegressorV1.refresh_readout

    def capture(self, *args):
        calls.append(self.kind)
        return original(self, *args)

    monkeypatch.setattr(models.SmoothLocalRegressorV1, "refresh_readout", capture)
    spec = next(v for v in SPECS if v["id"] == "hybrid_joint")
    result = fit_smooth_local(spec, arrays(), 0., .01, settings(), "cpu")
    assert result["status"] == "completed" and calls == ["joint"]


def test_failure_retains_initial_checkpoint_partial_state_and_does_not_retry(monkeypatch):
    count = []

    def fail(self, *args, **kwargs):
        count.append(1)
        raise FloatingPointError("injected nonfinite step")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    spec = next(v for v in SPECS if v["id"] == "hybrid_joint")
    result = fit_smooth_local(spec, arrays(), 0., .01, settings(), "cpu")
    assert result["status"] == "numerical_failure" and len(count) == 1
    assert len(result["checkpoints"]) == 1 and result["partial_state"] is not None
    assert "injected" in result["failures"][0]


def test_grid_truth_does_not_affect_selection_or_training():
    original = arrays()
    changed = copy.deepcopy(original)
    changed["grid_clean"] = 100 * changed["grid_clean"] + 3
    spec = next(v for v in SPECS if v["id"] == "hybrid_vp")
    results = [fit_smooth_local(spec, data, 0., .01, settings(), "cpu") for data in (original, changed)]
    assert results[0]["selected"] == results[1]["selected"]
    for left, right in zip(results[0]["checkpoints"], results[1]["checkpoints"]):
        assert left["validation_mse"] == right["validation_mse"]
        for key in left["state"]["state_dict"]:
            torch.testing.assert_close(left["state"]["state_dict"][key], right["state"]["state_dict"][key], atol=0, rtol=0)
    assert results[0]["metrics"] != results[1]["metrics"]


def test_bad_data_retains_failure_without_broadcasting():
    data = arrays()
    data["train_y"] = data["train_y"][:, 0]
    result = fit_smooth_local(SPECS[0], data, 0., None, settings(), "cpu")
    assert result["status"] == "failed"
    assert not result["checkpoints"] and result["partial_state"] is not None


def test_concentration_underflow_is_recorded_without_clipping(monkeypatch):
    def underflow(self, *args, **kwargs):
        with torch.no_grad():
            # Parameter order is readout weight/bias, then centers/raw widths.
            self.param_groups[0]["params"][-1].fill_(-1000.)

    monkeypatch.setattr(torch.optim.Adam, "step", underflow)
    spec = next(v for v in SPECS if v["id"] == "hybrid_joint")
    result = fit_smooth_local(spec, arrays(), 0., .01, settings(), "cpu")
    assert result["status"] == "numerical_failure"
    assert "positive" in result["failures"][0]
    assert torch.equal(result["partial_state"]["state_dict"]["basis.bank.raw_kappa"],
                       torch.full((10,), -1000., dtype=torch.float64))


def test_validation_ties_select_earliest_checkpoint():
    data = arrays()
    data["train_y"][:] = 0.
    data["validation_y"][:] = 0.
    spec = next(v for v in SPECS if v["id"] == "hybrid_joint")
    result = fit_smooth_local(spec, data, 0., .01, settings(), "cpu")
    assert result["status"] == "completed"
    assert [row["validation_mse"] for row in result["checkpoints"]] == [0., 0., 0.]
    assert set(result["selected"].values()) == {0}


def test_canonical_allowance_keys():
    assert budget_key(.25) == "0.25" and budget_key(1.) == "1"


def test_all_models_accept_real_task_pipeline_arrays_with_disposable_seed():
    from src.smooth_local_tasks_v1 import cell_arrays, generate_task, initializer_rotation

    config = copy.deepcopy(CONFIG)
    # Engineering integration does not consume a formal pilot/development or
    # confirmation function. Only this disposable root is generated here.
    config["seeds"]["pilot"] = 99103101
    config["seeds"]["initialization"] = 99103104
    task = generate_task(config, "pilot", "mixed")
    data = cell_arrays(task, 64)
    rotation = initializer_rotation(config, "pilot", "mixed", 0, 0)
    for spec in SPECS:
        result = fit_smooth_local(spec, data, rotation, .01 if spec["kind"] != "ridge" else None,
                                  settings(max_updates=2, validation_every=1), "cpu")
        assert result["status"] == "completed", (spec["id"], result["failures"])
        assert result["updates"] == (0 if spec["kind"] == "ridge" else 2)
        assert all(math.isfinite(row["normalized_mse"]) for row in result["metrics"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("model_id", ["fourier16", "hybrid_joint", "hybrid_vp", "line_rbf16"])
def test_tiny_cpu_cuda_training_parity(model_id):
    spec = next(v for v in SPECS if v["id"] == model_id)
    results = [fit_smooth_local(spec, arrays(), .3, .01, settings(), device) for device in ("cpu", "cuda")]
    assert all(result["status"] == "completed" for result in results)
    states = [result["checkpoints"][-1]["state"] for result in results]
    predictions = [predict_smooth_local_state(state, arrays()["grid_x"]) for state in states]
    # Joint Adam starts at a ridge stationary readout: float64 BLAS rounding
    # in its near-zero gradients is amplified by Adam's frozen epsilon=1e-8.
    # Check trajectory agreement on prediction scale, not bitwise equality.
    tolerance = 3e-8 if spec["kind"] == "joint" else 1e-10
    torch.testing.assert_close(*predictions, atol=tolerance, rtol=1e-9)
