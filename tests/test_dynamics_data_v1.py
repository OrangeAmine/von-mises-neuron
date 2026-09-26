"""Numerical, acquisition-time and split-causality tests for VM-N04 data."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.dynamics_data_v1 import (
    DynamicsConfig,
    aggregate_at_bin_end,
    clock_consistency_diagnostic,
    first_order_update,
    fit_normalizer,
    generate_trajectory,
    make_windows,
    sensor_history_at_origin,
    split_bounds,
    utc_daily_angles,
)


@pytest.fixture(scope="module")
def trajectory():
    return generate_trajectory(1107)


@pytest.mark.parametrize("ratio", [0.25, 1.0, 4.0])
def test_exact_update_matches_continuous_solution_and_semigroup(ratio):
    tau = 2.0 * ratio
    expected = 420.0 + 400.0 * (1 - np.exp(-2.0 / tau))
    value = 420.0
    for _ in range(8):
        value = first_order_update(value, 820.0, 0.25, tau)
    assert value == pytest.approx(expected, abs=5e-13)
    assert value == pytest.approx(first_order_update(420.0, 820.0, 2.0, tau), abs=5e-13)
    falling = first_order_update(value, 420.0, tau, tau)
    assert falling - 420 == pytest.approx((value - 420) / np.e)


@pytest.mark.parametrize("dt,tau", [(0, 1), (-1, 1), (1, 0), (1, np.inf), (np.nan, 1)])
def test_exact_update_rejects_invalid_time_constants(dt, tau):
    with pytest.raises(ValueError, match="positive and finite"):
        first_order_update(420.0, 800.0, dt, tau)


@pytest.mark.parametrize("ratio", [0.25, 1.0, 4.0])
def test_forcing_is_left_interval_value_not_premature_measurement(ratio):
    trajectory = generate_trajectory(12, DynamicsConfig(tau_ratio=ratio, noise_ppm=0))
    first_start = trajectory.metadata["daily_start_steps"][0]
    amplitude = trajectory.metadata["amplitude_ppm"]
    assert trajectory.latent_ppm[first_start] == 420.0
    assert trajectory.equilibrium_ppm[first_start] == 420 + amplitude
    expected = 420.0 + amplitude * (1.0 - np.exp(-2.0 / (2.0 * ratio)))
    assert trajectory.latent_ppm[first_start + 8] == pytest.approx(expected, abs=1e-12)
    assert trajectory.equilibrium_ppm[first_start + 8] == 420.0
    assert np.all(trajectory.latent_ppm >= 420.0)
    assert np.all(trajectory.latent_ppm <= 420.0 + amplitude)


def test_draws_are_reproducible_and_independent_across_rooms(trajectory):
    repeated = generate_trajectory(1107)
    other_room = generate_trajectory(1108)
    np.testing.assert_array_equal(repeated.observed_ppm, trajectory.observed_ppm)
    assert repeated.metadata == trajectory.metadata
    assert other_room.metadata["amplitude_ppm"] != trajectory.metadata["amplitude_ppm"]


def test_regime_randomness_coupled_and_shift_starts_only_in_test(trajectory):
    shifted = generate_trajectory(1107, DynamicsConfig(shifted=True, outage=True))
    irregular = generate_trajectory(1107, DynamicsConfig(irregular=True))
    test_start, _ = split_bounds(trajectory, "test")
    np.testing.assert_array_equal(shifted.observed_ppm[:test_start], trajectory.observed_ppm[:test_start])
    np.testing.assert_array_equal(shifted.event_times.asi8, trajectory.event_times.asi8)
    np.testing.assert_allclose(
        shifted.observed_ppm - shifted.latent_ppm,
        trajectory.observed_ppm - trajectory.latent_ppm, atol=1e-12,
    )
    np.testing.assert_allclose(
        irregular.observed_ppm - irregular.latent_ppm,
        trajectory.observed_ppm - trajectory.latent_ppm, atol=1e-12,
    )
    assert shifted.metadata["daily_presence_draw"] == irregular.metadata["daily_presence_draw"]
    starts = np.asarray(trajectory.metadata["daily_start_steps"])
    shifted_starts = np.asarray(shifted.metadata["daily_start_steps"])
    np.testing.assert_array_equal(shifted_starts[:42], starts[:42])
    np.testing.assert_array_equal(shifted_starts[42:] - starts[42:], 12)


def test_irregularity_is_realized_forcing_not_extra_input():
    trajectory = generate_trajectory(771, DynamicsConfig(irregular=True))
    daily_event_counts = trajectory.attendance.reshape(56, 96).sum(axis=1)
    assert set(daily_event_counts) == {0, 8}
    starts = np.asarray(trajectory.metadata["daily_start_steps"]) % 96
    nominal = trajectory.metadata["base_start_hour_utc"] * 4
    assert np.all(np.abs(starts - nominal) <= 2)
    batch = make_windows(trajectory, "test", fit_normalizer(trajectory))
    assert batch.sensor.shape[-1] == 3
    assert batch.future_angles.shape[-1] == 1


def test_outage_keeps_grid_scoring_truth_and_common_origins(trajectory):
    outage = generate_trajectory(1107, DynamicsConfig(outage=True))
    np.testing.assert_array_equal(outage.observed_ppm, trajectory.observed_ppm)
    np.testing.assert_array_equal(outage.event_times.asi8, trajectory.event_times.asi8)
    assert outage.outage.sum() == 3 * 8
    assert np.all(np.diff(outage.event_times.asi8) == pd.Timedelta(minutes=15).value)
    assert np.all(np.isfinite(outage.observed_ppm[outage.outage]))
    norm = fit_normalizer(outage)
    ordinary_batch = make_windows(trajectory, "test", norm)
    outage_batch = make_windows(outage, "test", norm)
    np.testing.assert_array_equal(ordinary_batch.origin_indices, outage_batch.origin_indices)
    np.testing.assert_array_equal(ordinary_batch.y_ppm, outage_batch.y_ppm)
    assert outage_batch.origin_outage.any()
    assert outage_batch.target_outage.any()
    assert outage_batch.exclusions["stale_or_missing_history"] == 0
    assert np.max(outage_batch.sensor[:, :, 2]) <= 1.0
    assert np.any(outage_batch.sensor[:, :, 2] > 1 / 12)


def test_outage_is_nominal_clock_based_even_without_attendance():
    trajectory = generate_trajectory(771, DynamicsConfig(shifted=True, irregular=True, outage=True))
    start = split_bounds(trajectory, "test")[0]
    nominal = int(trajectory.metadata["base_start_hour_utc"] * 4) + 12
    expected = np.zeros(len(trajectory.event_times), dtype=bool)
    for day in (2, 6, 10):
        left = start + day * 96 + nominal
        expected[left:left + 8] = True
    np.testing.assert_array_equal(expected, trajectory.outage)


def test_training_stats_ignore_future_unavailable_and_delayed_values(trajectory):
    altered = replace(trajectory, observed_ppm=trajectory.observed_ppm.copy())
    cutoff = split_bounds(trajectory, "train")[1]
    altered.observed_ppm[cutoff - 1:] = 1e12
    assert fit_normalizer(altered) == fit_normalizer(trajectory)
    available = trajectory.observed_ppm[:cutoff - 1]
    actual = fit_normalizer(trajectory)
    assert actual.mean == pytest.approx(available.mean())
    assert actual.std == pytest.approx(available.std())


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_shapes_dtypes_angles_timestamps_and_disjoint_splits(trajectory, split):
    batch = make_windows(trajectory, split, fit_normalizer(trajectory))
    n = len(batch.origin_indices)
    assert batch.sensor.shape == (n, 24, 3)
    assert batch.y.shape == (n, 16)
    assert batch.future_angles.shape == (n, 16, 1)
    assert batch.target_times.shape == (n, 16)
    assert batch.sensor.dtype == batch.y.dtype == batch.future_angles.dtype == np.float32
    assert batch.target_times.dtype == np.int64
    assert np.isfinite(batch.sensor).all() and np.isfinite(batch.y).all()
    assert str(batch.origin_times.tz) == "UTC"
    start, stop = split_bounds(trajectory, split)
    assert np.min(batch.origin_indices - 23) >= start
    assert np.max(batch.origin_indices + 16) < stop
    np.testing.assert_array_equal(batch.origin_indices % 4, 0)
    step_ns = pd.Timedelta(minutes=15).value
    np.testing.assert_array_equal(batch.target_times[:, 0] - batch.origin_times.asi8, step_ns)
    np.testing.assert_array_equal(batch.target_times[:, -1] - batch.origin_times.asi8, 16 * step_ns)
    target_indices = batch.origin_indices[:, None] + np.arange(1, 17)[None, :]
    np.testing.assert_array_equal(batch.future_angles, utc_daily_angles(trajectory.event_times)[target_indices].astype(np.float32))
    assert np.min(batch.future_angles) >= 0 and np.max(batch.future_angles) < 2 * np.pi
    accounted = sum(v for k, v in batch.exclusions.items() if k != "candidates")
    assert accounted == batch.exclusions["candidates"]
    if split != "test":
        assert np.all(trajectory.arrival_times[target_indices.ravel()] <= trajectory.event_times[stop - 1])


def test_label_arrival_cutoff_is_enforced_independently_of_event_time():
    trajectory = generate_trajectory(2, DynamicsConfig(split_days=(1, 1, 1), stride=1))
    norm = fit_normalizer(trajectory)
    batch = make_windows(trajectory, "train", norm)
    assert batch.exclusions["label_unavailable_by_cutoff"] == 1
    assert batch.origin_indices.max() + 16 == 94
    test = make_windows(trajectory, "test", norm)
    assert test.exclusions["label_unavailable_by_cutoff"] == 0
    assert test.origin_indices.max() + 16 == 287


def test_origin_sensor_is_delayed_but_previous_event_is_available(trajectory):
    norm = fit_normalizer(trajectory)
    history = sensor_history_at_origin(trajectory, 200, "train", norm)
    assert history is not None
    assert history[-1, 0] == pytest.approx(norm.transform(trajectory.observed_ppm[199]))
    assert history[-1, 1] == 0
    assert history[-1, 2] == pytest.approx(1 / 12)
    assert history[-2, 1] == 1
    assert history[-2, 2] == 0


def test_future_and_not_yet_arrived_historical_mutations_cannot_change_features(trajectory):
    norm = fit_normalizer(trajectory)
    origin = split_bounds(trajectory, "test")[0] + 100
    arrival_values = trajectory.arrival_times.asi8.copy()
    late_historical = origin - 10
    arrival_values[late_historical] = trajectory.event_times[origin + 1].value
    delayed = replace(trajectory, arrival_times=pd.to_datetime(arrival_values, utc=True))
    before = sensor_history_at_origin(delayed, origin, "test", norm)
    values = delayed.observed_ppm.copy()
    values[origin:] = -1e10
    values[late_historical] = 1e10
    after = sensor_history_at_origin(replace(delayed, observed_ppm=values), origin, "test", norm)
    np.testing.assert_array_equal(before, after)
    assert before[13, 1] == 0  # late_historical is the fourteenth history row.


def test_outage_truth_mutation_never_changes_features(trajectory):
    outage = generate_trajectory(1107, DynamicsConfig(outage=True))
    norm = fit_normalizer(outage)
    before = make_windows(outage, "test", norm)
    altered = replace(outage, observed_ppm=outage.observed_ppm.copy())
    altered.observed_ppm[altered.outage] = 9e9
    after = make_windows(altered, "test", norm)
    np.testing.assert_array_equal(before.sensor, after.sensor)
    np.testing.assert_array_equal(before.future_angles, after.future_angles)
    assert np.any(before.y_ppm != after.y_ppm)


def test_no_backward_fill_and_no_cross_split_fill(trajectory):
    norm = fit_normalizer(trajectory)
    origin = split_bounds(trajectory, "test")[0] + 100
    before = sensor_history_at_origin(trajectory, origin, "test", norm)
    changed = replace(trajectory, observed_ppm=trajectory.observed_ppm.copy())
    changed.observed_ppm[origin - 2] += 1000
    after = sensor_history_at_origin(changed, origin, "test", norm)
    np.testing.assert_array_equal(before[:-3], after[:-3])
    assert before[-3, 0] != after[-3, 0]
    start, _ = split_bounds(trajectory, "test")
    available = trajectory.sensor_available.copy()
    available[start:start + 24] = False
    unavailable = replace(trajectory, sensor_available=available)
    assert sensor_history_at_origin(unavailable, start + 23, "test", norm) is None


def test_stale_fill_rejects_whole_origin_and_reports_count(trajectory):
    cfg = replace(trajectory.config, max_fill_age_steps=2, outage=True)
    outage = generate_trajectory(1107, cfg)
    batch = make_windows(outage, "test", fit_normalizer(outage))
    assert batch.exclusions["stale_or_missing_history"] > 0
    assert np.max(batch.sensor[:, :, 2]) <= 1.0
    assert batch.exclusions["retained"] + sum(
        v for k, v in batch.exclusions.items() if k not in {"retained", "candidates"}
    ) == batch.exclusions["candidates"]


def test_removed_rows_and_irregular_grid_are_rejected(trajectory):
    with pytest.raises(ValueError, match="every regular"):
        fit_normalizer(replace(trajectory, event_times=trajectory.event_times.delete(100)))
    modified = trajectory.event_times.asi8.copy()
    modified[100] += pd.Timedelta(minutes=1).value
    with pytest.raises(ValueError, match="gap or irregular"):
        make_windows(replace(trajectory, event_times=pd.to_datetime(modified, utc=True)), "test", fit_normalizer(trajectory))


@pytest.mark.parametrize("date", ["2026-03-28T12:00Z", "2026-10-24T12:00Z"])
def test_clock_origin_split_and_dst_consistency(date):
    times = pd.date_range(date, periods=200, freq="15min")
    whole = utc_daily_angles(times)
    split = np.concatenate([utc_daily_angles(times[:83]), utc_daily_angles(times[83:])])
    np.testing.assert_array_equal(whole, split)
    np.testing.assert_array_equal(whole, utc_daily_angles(times.tz_convert("Europe/Budapest")))
    phase_step = np.angle(np.exp(1j * np.diff(whole[:, 0])))
    np.testing.assert_allclose(phase_step, 2 * np.pi / 96, atol=1e-14)
    diagnostic = clock_consistency_diagnostic(times)
    assert diagnostic["timezone_representation_max_error_rad"] == 0
    assert diagnostic["intentional_plus1h_defect_mean_error_rad"] == pytest.approx(np.pi / 12)


def test_daily_wrap_and_fixed_epoch():
    times = pd.to_datetime(["1969-12-31T23:45Z", "1970-01-01T00:00Z", "1970-01-01T06:00Z"], utc=True)
    np.testing.assert_allclose(utc_daily_angles(times)[:, 0], [2 * np.pi * 95 / 96, 0, np.pi / 2])
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_daily_angles(pd.date_range("2026-01-01", periods=2))


def test_aggregate_available_at_right_bin_edge_and_latest_arrival():
    times = pd.date_range("2026-01-01T00:00Z", periods=5, freq="15min")
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0], index=times)
    result = aggregate_at_bin_end(values)
    assert result.index[0] == pd.Timestamp("2026-01-01T01:00Z")
    assert result.iloc[0]["value"] == 2.5
    assert result.iloc[0]["count"] == 4
    assert result.iloc[0]["available_at"] == result.index[0]
    delayed = times.asi8.copy()
    delayed[3] = pd.Timestamp("2026-01-01T01:15Z").value
    delayed_result = aggregate_at_bin_end(values, pd.to_datetime(delayed, utc=True))
    assert delayed_result.iloc[0]["available_at"] == pd.Timestamp("2026-01-01T01:15Z")
    changed = values.copy()
    changed.iloc[-1] = -1e9
    pd.testing.assert_series_equal(aggregate_at_bin_end(changed).iloc[0], result.iloc[0])


def test_aggregate_retains_empty_grid_bins():
    times = pd.to_datetime(["2026-01-01T00:00Z", "2026-01-01T02:00Z"], utc=True)
    result = aggregate_at_bin_end(pd.Series([1.0, 2.0], index=times))
    assert len(result) == 3
    assert result.iloc[1]["count"] == 0
    assert np.isnan(result.iloc[1]["value"])
    assert pd.isna(result.iloc[1]["available_at"])
    assert result["value"].isna().all()  # No bin has all four required samples.
    assert not result["complete"].any()


@pytest.mark.parametrize("missing_kind", ["timestamp", "nan", "infinity", "no_arrival"])
def test_aggregate_requires_all_expected_samples(missing_kind):
    times = pd.date_range("2026-01-01T00:00Z", periods=4, freq="15min")
    observations = pd.Series([1.0, 2.0, 3.0, 4.0], index=times)
    arrivals = times.copy()
    if missing_kind == "timestamp":
        observations = observations.drop(times[2])
        arrivals = arrivals.delete(2)
    elif missing_kind == "no_arrival":
        arrival_values = arrivals.asi8.copy()
        arrival_values[2] = pd.NaT.value
        arrivals = pd.to_datetime(arrival_values, utc=True)
    else:
        observations.iloc[2] = np.nan if missing_kind == "nan" else np.inf
    row = aggregate_at_bin_end(observations, arrivals).iloc[0]
    assert row["count"] == 3
    assert row["expected_count"] == 4
    assert not row["complete"]
    assert np.isnan(row["value"])
    assert pd.isna(row["available_at"])


def test_aggregate_explicit_cadence_prevents_gap_inference():
    times = pd.date_range("2026-01-01T00:00Z", periods=3, freq="1min")
    values = pd.Series([2.0, 4.0, 6.0], index=times)
    result = aggregate_at_bin_end(values, interval="3min", sample_interval="1min")
    assert result.iloc[0]["complete"]
    assert result.iloc[0]["value"] == 4.0
    assert result.iloc[0]["available_at"] == pd.Timestamp("2026-01-01T00:03Z")
    missing = aggregate_at_bin_end(values.drop(times[1]), interval="3min", sample_interval="1min")
    assert not missing.iloc[0]["complete"]
    with pytest.raises(ValueError, match="divide"):
        aggregate_at_bin_end(values, interval="3min", sample_interval="2min")


@pytest.mark.parametrize("kwargs", [
    {"tau_ratio": 0}, {"noise_ppm": -1}, {"dt_minutes": 17},
    {"delay_steps": -1}, {"split_days": (1, 0, 1)},
    {"event_duration_hours": 0.3}, {"start_utc": "2026-01-01"},
    {"start_utc": "2026-01-01T01:00Z"}, {"max_fill_age_steps": 0},
])
def test_invalid_contract_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicsConfig(**kwargs)
