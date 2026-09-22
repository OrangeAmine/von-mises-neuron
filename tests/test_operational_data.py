"""Causality, alignment and cache policies for operational forecasting data."""

import numpy as np
import pandas as pd
import pytest

from src.operational_data import (
    OCCUPANCY_COLUMNS, POWER_COLUMNS, load_cached_hourly, prepare_forecast_windows,
)
from src.temporal import extract_angular_time_features


def frame_and_config():
    """Return a visible ramp and three half-open temporal segments."""
    index = pd.date_range("2024-01-01", periods=100, freq="1h")
    frame = pd.DataFrame({"sensor": np.arange(100, dtype=float),
                          "target": 100 + np.arange(100, dtype=float)}, index=index)
    config = dict(train_start=index[0], val_start=index[50], test_start=index[75],
                  test_end=index[-1] + pd.Timedelta(hours=1), lookback=8, horizon=4,
                  target_col="target")
    return frame, config


def test_normalization_train_only_and_raw_angle_order():
    frame, config = frame_and_config()
    splits, metadata = prepare_forecast_windows(frame, **config)
    changed = frame.copy()
    changed.iloc[50:] += 100000
    other, other_meta = prepare_forecast_windows(changed, **config)
    assert metadata["norm_stats"] == other_meta["norm_stats"]
    assert metadata["norm_stats"]["sensor"] == [24.5, np.std(np.arange(50))]
    np.testing.assert_array_equal(splits["train"]["X"], other["train"]["X"])
    np.testing.assert_array_equal(splits["train"]["y"], other["train"]["y"])
    for data in splits.values():
        assert data["X"].dtype == np.float32
        assert data["y"].dtype == np.float32
        np.testing.assert_array_equal(data["X_sensor"], data["X"][..., :2])
        angles = extract_angular_time_features(pd.DatetimeIndex(data["input_times"].ravel()), [24, 168])
        np.testing.assert_allclose(data["X"][..., -2:], angles.reshape(-1, 8, 2), atol=3e-7)


def test_disjoint_targets_and_shared_causal_context():
    frame, config = frame_and_config()
    splits, metadata = prepare_forecast_windows(frame, **config)
    assert metadata["splits"]["train"]["valid_windows"] == 39
    assert metadata["splits"]["val"]["valid_windows"] == 22
    assert metadata["splits"]["test"]["valid_windows"] == 22
    for split, start in [("val", config["val_start"]), ("test", config["test_start"])]:
        data = splits[split]
        assert data["target_times"][0, 0] == start
        assert data["origins"][0] == start - pd.Timedelta(hours=1)
        assert data["input_times"][0, 0] == start - pd.Timedelta(hours=8)
    target_sets = [set(data["target_times"].ravel()) for data in splits.values()]
    assert not target_sets[0] & target_sets[1]
    assert not target_sets[1] & target_sets[2]
    for data in splits.values():
        assert (data["input_times"] <= data["origins"][:, None]).all()
        assert (data["target_times"] > data["origins"][:, None]).all()
        assert np.all(np.diff(data["target_times"], axis=1) == np.timedelta64(1, "h"))
        expected = frame.loc[pd.DatetimeIndex(data["target_times"].ravel()), "target"]
        np.testing.assert_array_equal(data["y_raw"], expected.to_numpy().reshape(-1, 4))


@pytest.mark.parametrize("missing_kind", ["drop", "nan", "inf"])
def test_missing_hour_rejects_crossing_windows(missing_kind):
    frame, config = frame_and_config()
    missing = frame.index[60]
    if missing_kind == "drop":
        frame = frame.drop(missing)
    else:
        frame.loc[missing] = np.nan if missing_kind == "nan" else np.inf
    splits, metadata = prepare_forecast_windows(frame, **config)
    data = splits["val"]
    assert missing not in data["input_times"]
    assert missing not in data["target_times"]
    assert metadata["splits"]["val"]["valid_windows"] == 10
    assert metadata["splits"]["val"]["nonfinite_input_windows"] == 8
    assert metadata["splits"]["val"]["nonfinite_target_windows"] == 4
    assert metadata["missing_hour_rows_inserted"] == (1 if missing_kind == "drop" else 0)


def test_future_targets_and_covariates_cannot_change_origin_inputs():
    frame, config = frame_and_config()
    original, _ = prepare_forecast_windows(frame, **config)
    changed = frame.copy()
    changed.iloc[75:, 0] = np.nan
    changed.iloc[75:, 1] += 1000
    future, _ = prepare_forecast_windows(changed, **config)
    # The first test forecast is still possible using only earlier observations.
    np.testing.assert_array_equal(original["test"]["X"][0], future["test"]["X"][0])
    np.testing.assert_array_equal(original["val"]["X"], future["val"]["X"])
    assert len(future["test"]["X"]) == 1
    np.testing.assert_array_equal(future["test"]["y_raw"][0], original["test"]["y_raw"][0] + 1000)


def test_empty_split_shapes_and_constant_training_sensor():
    frame, config = frame_and_config()
    frame["sensor"] = 5
    frame.loc[frame.index[50]:, "target"] = np.nan
    splits, metadata = prepare_forecast_windows(frame, **config)
    assert metadata["norm_stats"]["sensor"] == [5, 1]
    assert splits["val"]["X"].shape == (0, 8, 4)
    assert splits["val"]["y"].shape == (0, 4)
    assert splits["val"]["target_times"].shape == (0, 4)
    assert metadata["splits"]["val"]["pooled_window_targets"]["mean"] is None


@pytest.mark.parametrize("error", ["duplicate", "unordered", "fractional", "boundary", "size"])
def test_rejects_invalid_window_contract(error):
    frame, config = frame_and_config()
    if error == "duplicate":
        frame = pd.concat([frame, frame.iloc[-1:]])
    elif error == "unordered":
        frame = frame.iloc[::-1]
    elif error == "fractional":
        frame.index += pd.Timedelta(minutes=1)
    elif error == "boundary":
        config["val_start"] = config["test_start"]
    else:
        config["lookback"] = 0
    with pytest.raises(ValueError):
        prepare_forecast_windows(frame, **config)


def write_power(tmp_path, rows):
    """Write a small UCI-format cache without network calls."""
    directory = tmp_path / "power" / "extracted"
    directory.mkdir(parents=True)
    path = directory / "household_power_consumption.txt"
    pd.DataFrame(rows, columns=["Date", "Time"] + POWER_COLUMNS).to_csv(path, sep=";", index=False)
    return path


def test_power_right_edge_means_missing_grid_and_fingerprints(tmp_path):
    path = write_power(tmp_path, [
        ["01/01/2024", "00:00:00"] + [1] * 7,
        ["01/01/2024", "00:59:00"] + [3] * 7,
        ["01/01/2024", "01:00:00"] + [10] * 7,
        ["01/01/2024", "03:00:00"] + ["?"] * 7,
        ["01/01/2024", "04:00:00"] + [5] * 7,
    ])
    before = path.read_bytes()
    frame, metadata = load_cached_hourly("power", tmp_path)
    assert len(frame) == 5
    assert frame.index[0] == pd.Timestamp("2024-01-01 01:00")
    assert frame.iloc[0, 0] == 2
    assert frame.iloc[1, 0] == 10
    assert frame.iloc[2:4].isna().all().all()
    assert metadata["all_features_missing_hours_before_coverage"] == 2
    assert metadata["raw_files"][0]["bytes"] == len(before)
    assert len(metadata["raw_files"][0]["sha256"]) == 64
    assert path.read_bytes() == before


def test_power_coverage_is_per_sensor_and_unfilled(tmp_path):
    rows = [["01/01/2024", f"00:{minute:02d}:00"] + [1] * 7 for minute in range(45)]
    rows[-1][-1] = "?"
    write_power(tmp_path, rows)
    frame, metadata = load_cached_hourly("power", tmp_path, min_coverage=.75)
    assert frame.iloc[0, :-1].notna().all()
    assert pd.isna(frame.iloc[0, -1])
    assert metadata["minimum_readings_per_feature"] == 45


def write_occupancy(tmp_path, conflict=False):
    """Create repeated sensor rows in the three-file cache convention."""
    directory = tmp_path / "occupancy" / "extracted"
    directory.mkdir(parents=True)
    first = pd.DataFrame([["2024-01-01 00:00:00"] + [1] * 5], columns=["date"] + OCCUPANCY_COLUMNS)
    first.to_csv(directory / "datatraining.txt", index=False)
    second = first.copy()
    if conflict:
        second.loc[0, "CO2"] = 2
    second.to_csv(directory / "datatest.txt", index=False)


def test_occupancy_duplicate_policy(tmp_path):
    write_occupancy(tmp_path)
    frame, metadata = load_cached_hourly("occupancy", tmp_path)
    assert len(frame) == 1
    assert metadata["identical_duplicate_rows_removed"] == 1
    assert metadata["raw_rows"] == 2
    assert list(frame.columns) == OCCUPANCY_COLUMNS
    assert frame.index[0] == pd.Timestamp("2024-01-01 01:00:00")


def test_occupancy_rejects_conflicting_duplicates(tmp_path):
    write_occupancy(tmp_path, conflict=True)
    with pytest.raises(ValueError, match="Conflicting sensor duplicates"):
        load_cached_hourly("occupancy", tmp_path)


def test_missing_cache_fails_without_creating_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cached_hourly("power", tmp_path)
    assert not list(tmp_path.iterdir())
