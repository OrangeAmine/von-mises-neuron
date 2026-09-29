"""Disposable fixture evidence for VM-N02 generation and existing-model fits."""

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.circular_layers_v2 import CartesianCircularLinearV1
from src.interpretation_n02_cases import (
    _entropy, fit_peak, fit_proxy, make_peak_unit, make_proxy_unit,
)
from src.locality_models import circular_features, predict_readout


@pytest.fixture
def config():
    """Use only disposable namespace and shorter engineering-fit budgets."""
    path = Path(__file__).resolve().parents[1] / "configs" / "vm_n02_interpretation_v1.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result["samples"] = {"train": 64, "validation": 32, "evaluation": 64, "curve": 128}
    result["peak"]["steps"] = 50
    return result


@pytest.mark.parametrize("case", ["isolated", "redundant"])
def test_peak_equation_dtypes_shapes_and_reproducibility(config, case):
    torch_state = torch.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    unit = make_peak_unit(config, "fixture", 0, case)
    again = make_peak_unit(config, "fixture", 0, case)
    assert torch.equal(torch_state, torch.get_rng_state())
    np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
    truth = unit["truth"]
    assert truth["centers"].shape == ((1,) if case == "isolated" else (2,))
    for split in ("train", "validation", "evaluation", "curve"):
        values = unit[split]
        assert values["x"].shape == values["clean"].shape == (config["samples"][split],)
        reference = predict_readout(circular_features(values["x"], truth["centers"], truth["kappa"]),
                                    truth["weights"], truth["bias"])
        torch.testing.assert_close(reference, values["clean"], atol=1e-14, rtol=1e-14)
        for name, tensor in values.items():
            assert tensor.dtype == torch.float64 and tensor.device.type == "cpu"
            assert not tensor.requires_grad and bool(torch.isfinite(tensor).all())
            torch.testing.assert_close(tensor, again[split][name], atol=0, rtol=0)
    assert not torch.equal(unit["train"]["x"][:32], unit["validation"]["x"])
    assert not torch.equal(unit["train"]["y"], unit["train"]["clean"])
    assert torch.unique(unit["curve"]["x"]).numel() == config["samples"]["curve"]
    another = make_peak_unit(config, "fixture", 1, case)
    assert not torch.equal(truth["centers"], another["truth"]["centers"])


def test_disjoint_addresses_and_named_observation_streams(config):
    addresses = {_entropy(config, stage, 0, case)
                 for stage in ("fixture", "pilot", "development", "evaluation")
                 for case in ("isolated", "redundant", "proxy")}
    assert len(addresses) == 12
    unit = make_peak_unit(config, "fixture", 0, "isolated")
    entropy = unit["seed_components"]
    assert entropy[1] == 40000
    noise = np.random.default_rng(np.random.SeedSequence((*entropy, 2))).normal(
        0, config["peak"]["noise_sd"], config["samples"]["train"])
    torch.testing.assert_close(unit["train"]["y"] - unit["train"]["clean"],
                               torch.tensor(noise), atol=1e-15, rtol=1e-14)
    modified = copy.deepcopy(config)
    modified["samples"]["train"] = 71
    other = make_peak_unit(modified, "fixture", 0, "isolated")
    # Changing one observation pool's size cannot consume another pool's RNG.
    torch.testing.assert_close(unit["validation"]["y"], other["validation"]["y"], atol=0, rtol=0)


@pytest.mark.parametrize("stage,index,case", [
    ("fixture", 2, "isolated"), ("fixture", -1, "isolated"),
    ("fixture", True, "isolated"), ("unknown", 0, "isolated"),
    ("fixture", 0, "proxy"),
])
def test_invalid_peak_addresses_rejected(config, stage, index, case):
    with pytest.raises(ValueError):
        make_peak_unit(config, stage, index, case)


def test_proxy_generator_pairing_and_counterfactual(config):
    unit = make_proxy_unit(config, "fixture", 0)
    another = make_proxy_unit(config, "fixture", 1)
    assert not torch.equal(unit["truth"]["center"], another["truth"]["center"])
    truth = unit["truth"]
    aligned, shifted = (unit["evaluation"][name] for name in ("aligned", "shifted"))
    delta = config["proxy"]["shift_radians"]
    for split in (unit["train"], unit["validation"], aligned, shifted):
        assert split["x"].ndim == 2 and split["x"].shape[1] == 2
        assert split["x"].dtype == split["clean"].dtype == torch.float64
        reference = truth["bias"] + truth["amplitude"] * (split["latent"] - truth["center"]).cos()
        torch.testing.assert_close(reference, split["clean"], atol=2e-15, rtol=2e-15)
        # A surgical recorded-clock change leaves latent generator truth fixed.
        changed_x = split["x"].clone()
        changed_x[:, 1] += math.pi / 2
        assert not torch.equal(changed_x, split["x"])
        after = truth["bias"] + truth["amplitude"] * (split["latent"] - truth["center"]).cos()
        torch.testing.assert_close(after, reference, atol=0, rtol=0)
    torch.testing.assert_close(aligned["x"][:, 1], shifted["x"][:, 1], atol=0, rtol=0)
    torch.testing.assert_close(shifted["latent"] - aligned["latent"],
                               torch.full_like(aligned["latent"], delta), atol=1e-14, rtol=0)
    torch.testing.assert_close(shifted["x"][:, 0] - shifted["latent"],
                               aligned["x"][:, 0] - aligned["latent"], atol=1e-14, rtol=0)
    torch.testing.assert_close(shifted["y"] - shifted["clean"],
                               aligned["y"] - aligned["clean"], atol=3e-16, rtol=0)
    assert not torch.equal(unit["train"]["latent"][:32], unit["validation"]["latent"])


@pytest.mark.parametrize("case", ["isolated", "redundant"])
def test_peak_fit_checkpoint_replay_parameter_count_and_no_clean_label_access(config, case):
    unit = make_peak_unit(config, "fixture", 0, case)
    result = fit_peak(unit, config, 11)
    expected_step = min(result["validation_history"], key=lambda row: row["validation_mse"])["step"]
    assert result["selected_step"] == expected_step
    assert [row["step"] for row in result["validation_history"]] == [0, 25, 50]
    assert result["parameter_count"] == 3 * config["peak"]["kernels"][case] + 1
    assert result["seconds"] > 0 and result["init_seed"] == 11
    for key in ("centers", "kappa", "weights", "bias"):
        tensor = result[key]
        assert tensor.dtype == torch.float64 and tensor.device.type == "cpu"
        assert not tensor.requires_grad and bool(torch.isfinite(tensor).all())
    prediction = predict_readout(circular_features(unit["validation"]["x"], result["centers"], result["kappa"]),
                                 result["weights"], result["bias"])
    assert float((prediction - unit["validation"]["y"]).square().mean()) == pytest.approx(
        result["validation_mse"], abs=1e-14)
    poisoned = {key: unit[key] for key in ("train", "validation", "case", "seed_components")}
    poisoned["train"] = {key: unit["train"][key] for key in ("x", "y")}
    poisoned["validation"] = {key: unit["validation"][key] for key in ("x", "y")}
    repeated = fit_peak(poisoned, config, 11)
    for key in ("centers", "kappa", "weights", "bias"):
        torch.testing.assert_close(result[key], repeated[key], atol=0, rtol=0)


def test_isolated_initialization_uses_observed_training_maximum(config):
    unit = make_peak_unit(config, "fixture", 0, "isolated")
    unit["train"]["y"] = torch.zeros_like(unit["train"]["y"])
    unit["train"]["y"][4] = 100
    result = fit_peak(unit, config, 29)
    distance = abs(float(result["initial_state"]["centers"][0] - unit["train"]["x"][4]))
    assert distance <= config["peak"]["isolated_center_jitter"]


def test_proxy_readout_matches_existing_cartesian_api_and_ridge_stationarity(config):
    unit = make_proxy_unit(config, "fixture", 0)
    state = fit_proxy(unit, config)
    # Constructor RNG is isolated; explicitly loaded coefficients carry the fit.
    with torch.random.fork_rng():
        layer = CartesianCircularLinearV1(2, 1, dtype=torch.float64)
    with torch.no_grad():
        layer.weight_cos.copy_(state["weights"][:2][None, :])
        layer.weight_sin.copy_(state["weights"][2:][None, :])
        layer.bias.copy_(state["bias"].reshape(1))
    for split in (unit["train"], unit["validation"], unit["evaluation"]["shifted"]):
        features = torch.cat((split["x"].cos(), split["x"].sin()), dim=1)
        expected = features @ state["weights"] + state["bias"]
        torch.testing.assert_close(layer(split["x"])[:, 0], expected, atol=1e-14, rtol=1e-14)
    features = torch.cat((unit["train"]["x"].cos(), unit["train"]["x"].sin()), dim=1)
    for design, weight, bias in ((features, state["weights"], state["bias"]),
                                 (features[:, [0, 2]], state["sensor_weights"], state["sensor_bias"])):
        residual = design @ weight + bias - unit["train"]["y"]
        gradient = design.T @ residual / len(design) + config["proxy"]["ridge"] * weight
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-15, rtol=0)
        assert abs(float(residual.mean())) < 2e-15
    assert state["parameter_count"] == 5 and state["sensor_parameter_count"] == 3
    minimal = {"train": {key: unit["train"][key] for key in ("x", "y")}}
    again = fit_proxy(minimal, config)
    for key in ("weights", "bias", "sensor_weights", "sensor_bias"):
        torch.testing.assert_close(state[key], again[key], atol=0, rtol=0)


@pytest.mark.parametrize("case", ["isolated", "proxy"])
def test_deadline_and_nonfinite_failures_propagate(config, case):
    unit = (make_proxy_unit(config, "fixture", 0) if case == "proxy" else
            make_peak_unit(config, "fixture", 0, case))
    def run(deadline=None):
        return (fit_proxy(unit, config, deadline=deadline) if case == "proxy" else
                fit_peak(unit, config, 11, deadline=deadline))
    with pytest.raises(TimeoutError, match="deadline"):
        run(time.perf_counter() - 1)
    unit["train"]["y"][0] = math.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run()


def test_fit_rejects_wrong_dtype_and_invalid_step_schedule(config):
    unit = make_peak_unit(config, "fixture", 0, "isolated")
    wrong_dtype = copy.deepcopy(unit)
    wrong_dtype["train"]["x"] = wrong_dtype["train"]["x"].float()
    with pytest.raises(ValueError, match="float64"):
        fit_peak(wrong_dtype, config, 11)
    config["peak"]["steps"] = 24
    with pytest.raises(ValueError, match="divisible"):
        fit_peak(unit, config, 11)
