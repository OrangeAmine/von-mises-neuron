"""Meaningful checks for cached-data diagnosis and causal simple controls."""

import numpy as np
import pandas as pd
import pytest

from scripts.diagnose_occupancy import diagnose, evaluate_split, original_unit_metrics


def test_daily_seasonal_control_reconstructs_hourly_periodic_signal():
    index = pd.date_range("2025-01-01", periods=60, freq="h")
    frame = pd.DataFrame({"CO2": 600 + np.sin(np.arange(60) * 2 * np.pi / 24)}, index=index)
    report = evaluate_split(frame, train_mean=600)
    common = report["common_causal_windows"]
    assert common["windows"] == 31
    assert common["metrics"]["seasonal_24h"]["rmse"] < 1e-12
    assert common["metrics"]["persistence"]["rmse"] > 0
    assert report["windows_crossing_time_gaps"] == 0
    assert report["common_contiguous_hourly_windows"] == common


def test_persistence_errors_use_last_observation_at_every_horizon():
    frame = pd.DataFrame({"CO2": np.arange(40.) + 500},
                         index=pd.date_range("2025-01-01", periods=40, freq="h"))
    report = evaluate_split(frame, train_mean=500)
    metrics = report["common_causal_windows"]["metrics"]["persistence"]
    assert metrics["mae"] == pytest.approx(3.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(np.arange(1., 7.) ** 2)))
    assert report["common_causal_windows"]["target_summary"]["count"] == 11 * 6


def test_gap_windows_are_recorded_and_seasonal_sources_must_precede_origin():
    index = pd.date_range("2025-01-01", periods=100, freq="h").delete(slice(24, 54))
    frame = pd.DataFrame({"CO2": np.arange(len(index), dtype=float) + 500}, index=index)
    report = evaluate_split(frame, train_mean=500)
    assert report["windows_crossing_time_gaps"] > 0
    assert report["last_horizon_elapsed_hours"]["max"] == 36
    assert report["common_causal_windows"]["windows"] < report["finite_windows"]
    # A 30h gap makes t-24h unavailable/future for the first target block.
    assert report["common_causal_windows"]["first_origin_bin"] > str(index[23])


def test_short_season_excludes_future_target_sources():
    index = pd.date_range("2025-01-01", periods=40, freq="h")
    frame = pd.DataFrame({"CO2": np.arange(40.) + 500}, index=index)
    report = evaluate_split(frame, train_mean=500, season_hours=2)
    assert report["finite_windows"] == 11
    assert report["common_causal_windows"]["windows"] == 0


def test_all_controls_share_nan_filtered_windows():
    index = pd.date_range("2025-01-01", periods=80, freq="h")
    frame = pd.DataFrame({"CO2": np.arange(80.) + 500, "sensor": np.ones(80)}, index=index)
    frame.loc[index[35], "sensor"] = np.nan
    report = evaluate_split(frame, train_mean=500)
    assert report["legacy_nan_valid_windows"] == 27
    common = report["common_causal_windows"]
    assert common["windows"] == 27
    assert common["target_summary"]["count"] == 27 * 6
    assert len(common["metrics"]) == 3


def test_empty_windows_and_constant_target_are_not_reported_as_valid_r2():
    frame = pd.DataFrame({"CO2": [600.] * 4},
                         index=pd.date_range("2025-01-01", periods=4, freq="h"))
    assert evaluate_split(frame, train_mean=600)["common_causal_windows"]["windows"] == 0
    assert original_unit_metrics(np.ones((2, 2)), np.ones((2, 2)))["r2"] is None


def test_missing_cache_fails_before_loader_or_download(monkeypatch, tmp_path):
    def forbidden_loader(**kwargs):
        raise AssertionError("Loader must not run for absent cache")
    monkeypatch.setattr("scripts.diagnose_occupancy.load_occupancy", forbidden_loader)
    with pytest.raises(FileNotFoundError, match="downloads disabled"):
        diagnose(tmp_path)
