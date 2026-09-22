"""Check clock correction and checkpoint reproduction without neural training."""

import numpy as np
import pytest
import torch

from scripts.benchmark_standard_dataset import generate_synthetic_co2
from scripts.evaluate_synthetic_clock_sensitivity import (
    FIXED_CLOCK, clock_window_groups, prepare_clock_pair, require_reproduction,
)
from src.temporal import extract_angular_time_features


@pytest.fixture(scope="module")
def frame():
    return generate_synthetic_co2(n_days=90, seed=42)


def test_fixed_offset_angles_preserve_all_fitting_data(frame):
    original = extract_angular_time_features(frame.index, [24, 168])
    corrected = extract_angular_time_features(frame.index.tz_convert(FIXED_CLOCK), [24, 168])
    np.testing.assert_array_equal(original[:1944], corrected[:1944])
    np.testing.assert_array_equal(original[:2066], corrected[:2066])
    post_delta = np.angle(np.exp(1j * (original[2066:] - corrected[2066:])))
    np.testing.assert_allclose(post_delta, np.tile([2*np.pi/24, 2*np.pi/168], (94, 1)), atol=1e-12)


def test_corrected_phase_is_coherent_with_elapsed_generator(frame):
    corrected = extract_angular_time_features(frame.index.tz_convert(FIXED_CLOCK), [24, 168])
    elapsed = np.arange(len(frame))
    for column, period in enumerate([24, 168]):
        delta = corrected[:, column] - 2 * np.pi * elapsed / period
        np.testing.assert_allclose(np.exp(1j*delta), np.full(len(frame), np.exp(1j*delta[0])), atol=1e-12)


def test_mismatch_window_counts_and_disjoint_target_groups(frame):
    groups, counts = clock_window_groups(frame.index[1944:])
    assert counts == {
        "all": 133, "pre_transition_targets": 39, "straddling_targets": 11,
        "post_transition_targets": 83, "changed_test_rows": 94,
        "changed_target_entries": 1062, "target_entries": 1596,
        "inputs_with_changed_clock": 82, "inputs_entirely_changed_clock": 11,
    }
    np.testing.assert_array_equal(sum(groups[name].astype(int) for name in groups if name != "all"), np.ones(133))


def test_clock_pair_changes_only_test_angles_not_sensors_or_targets(frame):
    original, corrected, hashes = prepare_clock_pair(frame, torch.device("cpu"))
    for split in ["train", "val", "test"]:
        assert torch.equal(original[f"y_{split}"], corrected[f"y_{split}"])
        assert torch.equal(original[f"X_{split}"][..., :4], corrected[f"X_{split}"][..., :4])
    assert hashes["train"]["original_input"] == hashes["train"]["corrected_input"]
    assert hashes["val"]["original_input"] == hashes["val"]["corrected_input"]
    assert hashes["test"]["original_input"] != hashes["test"]["corrected_input"]
    changed_inputs = (original["X_test"] != corrected["X_test"]).any(dim=2).any(dim=1)
    assert changed_inputs.sum().item() == 82


def test_saved_prediction_reproduction_accepts_exact_and_rejects_changed_checkpoint(tmp_path):
    prediction = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    target = torch.ones_like(prediction)
    artifact = tmp_path / "predictions.npz"
    np.savez(artifact, prediction=prediction.numpy(), target=target.numpy())
    assert require_reproduction(prediction, target, artifact) == 0
    with pytest.raises(ValueError, match="failed reproduction"):
        require_reproduction(prediction + 0.01, target, artifact)
    with pytest.raises(ValueError, match="targets do not match"):
        require_reproduction(prediction, target + 1, artifact)


def test_reproduction_rejects_nonfinite_saved_predictions(tmp_path):
    artifact = tmp_path / "predictions.npz"
    prediction = torch.zeros((2, 2))
    np.savez(artifact, prediction=np.full((2, 2), np.nan), target=prediction.numpy())
    with pytest.raises(ValueError, match="Invalid original predictions"):
        require_reproduction(prediction, prediction, artifact)
