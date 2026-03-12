"""Tests for src/data_loading.py -- dataset registry, preprocessing logic,
and end-to-end pipeline validation.

All tests use synthetic DataFrames that mimic the real dataset schemas.
No actual downloads are triggered -- we test the preprocessing/pipeline
logic in isolation from the network.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loading import (
    DATASET_CONFIGS,
    list_datasets,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Helpers: synthetic DataFrames that mimic real dataset schemas
# ---------------------------------------------------------------------------
def _make_occupancy_df(n_rows: int = 500) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking UCI Occupancy after resampling.

    Columns: Temperature, Humidity, Light, CO2, HumidityRatio
    Index: hourly DatetimeIndex
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2015-02-04", periods=n_rows, freq="1h")
    t = np.arange(n_rows, dtype=float)
    return pd.DataFrame(
        {
            "Temperature": 22.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, n_rows),
            "Humidity": 28.0 + 4.0 * np.sin(2 * np.pi * t / 24 + 1.0) + rng.normal(0, 0.5, n_rows),
            "Light": np.clip(200.0 * np.sin(2 * np.pi * t / 24) + 250, 0, 500) + rng.normal(0, 10, n_rows),
            "CO2": 600.0 + 80.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 10, n_rows),
            "HumidityRatio": 0.004 + 0.001 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.0002, n_rows),
        },
        index=idx,
    )


def _make_power_df(n_rows: int = 2000) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking UCI Power after resampling.

    Columns: 7 power-related features at hourly resolution.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2006-12-16", periods=n_rows, freq="1h")
    t = np.arange(n_rows, dtype=float)
    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / 168)
    return pd.DataFrame(
        {
            "Global_active_power": 1.0 + 0.8 * daily + 0.3 * weekly + rng.normal(0, 0.2, n_rows),
            "Global_reactive_power": 0.1 + 0.05 * daily + rng.normal(0, 0.02, n_rows),
            "Voltage": 240.0 + 5.0 * daily + rng.normal(0, 1.0, n_rows),
            "Global_intensity": 4.0 + 3.0 * daily + rng.normal(0, 0.5, n_rows),
            "Sub_metering_1": np.clip(rng.exponential(0.5, n_rows), 0, 20),
            "Sub_metering_2": np.clip(rng.exponential(1.0, n_rows), 0, 30),
            "Sub_metering_3": 5.0 + 3.0 * daily + rng.normal(0, 1.0, n_rows),
        },
        index=idx,
    )


def _make_jena_df(n_rows: int = 3000) -> pd.DataFrame:
    """Create a synthetic DataFrame mimicking Jena Climate after resampling.

    Columns: T (degC), p (mbar), rh (%), wv (m/s), wd (deg)
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2009-01-01", periods=n_rows, freq="1h")
    t = np.arange(n_rows, dtype=float)
    daily = np.sin(2 * np.pi * t / 24)
    annual = np.sin(2 * np.pi * t / 8766)
    return pd.DataFrame(
        {
            "T (degC)": 10.0 + 5.0 * daily + 10.0 * annual + rng.normal(0, 1.5, n_rows),
            "p (mbar)": 1013.0 + 5.0 * annual + rng.normal(0, 2.0, n_rows),
            "rh (%)": 70.0 - 10.0 * daily + rng.normal(0, 5.0, n_rows),
            "wv (m/s)": np.clip(3.0 + 1.5 * daily + rng.normal(0, 0.8, n_rows), 0, None),
            "wd (deg)": rng.uniform(0, 360, n_rows),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Dataset registry tests
# ---------------------------------------------------------------------------
class TestDatasetRegistry:
    """Tests for the DATASET_CONFIGS registry and lookup functions."""

    def test_registry_has_three_datasets(self):
        """Registry contains exactly occupancy, power, jena."""
        assert set(DATASET_CONFIGS.keys()) == {"occupancy", "power", "jena"}

    def test_list_datasets_sorted(self):
        """list_datasets() returns sorted names."""
        names = list_datasets()
        assert names == sorted(names)
        assert len(names) == 3

    def test_each_config_has_required_keys(self):
        """Every dataset config contains the mandatory keys."""
        required = {
            "loader", "target_col", "unit", "lookback", "horizon",
            "handcrafted_periods", "split_ratios", "description",
        }
        for name, config in DATASET_CONFIGS.items():
            missing = required - set(config.keys())
            assert not missing, f"{name} missing keys: {missing}"

    def test_loaders_are_callable(self):
        """Each loader entry is actually callable."""
        for name, config in DATASET_CONFIGS.items():
            assert callable(config["loader"]), f"{name}.loader is not callable"

    def test_split_ratios_sum_to_one(self):
        """Split ratios for each dataset sum to 1.0."""
        for name, config in DATASET_CONFIGS.items():
            total = sum(config["split_ratios"])
            assert abs(total - 1.0) < 1e-6, (
                f"{name}: split_ratios sum to {total}, not 1.0"
            )

    def test_invalid_dataset_raises(self):
        """load_dataset raises ValueError for unknown names."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("nonexistent_dataset")

    def test_occupancy_reduced_params(self):
        """Occupancy uses smaller lookback/horizon due to limited data."""
        cfg = DATASET_CONFIGS["occupancy"]
        # Only ~432 hourly rows -- must use small windows
        assert cfg["lookback"] <= 48
        assert cfg["horizon"] <= 12
        # Wider val/test split to get enough sequences
        assert cfg["split_ratios"][1] >= 0.15

    def test_handcrafted_periods_positive(self):
        """All handcrafted periods are positive integers."""
        for name, config in DATASET_CONFIGS.items():
            for p in config["handcrafted_periods"]:
                assert isinstance(p, int) and p > 0, (
                    f"{name}: invalid period {p}"
                )


# ---------------------------------------------------------------------------
# Occupancy preprocessing logic tests
# ---------------------------------------------------------------------------
class TestOccupancyPreprocessing:
    """Tests for occupancy-specific preprocessing logic."""

    def test_schema_columns(self):
        """Synthetic occupancy df has the expected columns."""
        df = _make_occupancy_df()
        expected_cols = {"Temperature", "Humidity", "Light", "CO2", "HumidityRatio"}
        assert set(df.columns) == expected_cols

    def test_no_occupancy_column(self):
        """Occupancy column should NOT be present (it's a label, not a feature)."""
        df = _make_occupancy_df()
        assert "Occupancy" not in df.columns

    def test_hourly_resolution(self):
        """Index is at hourly resolution."""
        df = _make_occupancy_df()
        deltas = pd.Series(df.index).diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()

    def test_no_nans(self):
        """Synthetic data has no NaN values."""
        df = _make_occupancy_df()
        assert df.isna().sum().sum() == 0

    def test_co2_in_realistic_range(self):
        """CO2 values are in a plausible indoor range."""
        df = _make_occupancy_df()
        assert df["CO2"].min() > 200
        assert df["CO2"].max() < 2000


# ---------------------------------------------------------------------------
# Power consumption preprocessing logic tests
# ---------------------------------------------------------------------------
class TestPowerPreprocessing:
    """Tests for power consumption preprocessing logic."""

    def test_schema_columns(self):
        """Synthetic power df has the expected 7 columns."""
        df = _make_power_df()
        assert len(df.columns) == 7
        assert "Global_active_power" in df.columns

    def test_no_question_marks(self):
        """No string '?' values in the data (they should be NaN or removed)."""
        df = _make_power_df()
        for col in df.columns:
            assert not (df[col].astype(str) == "?").any()

    def test_all_numeric(self):
        """All columns are numeric dtype."""
        df = _make_power_df()
        for col in df.columns:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_hourly_resolution(self):
        """Index is at hourly resolution."""
        df = _make_power_df()
        deltas = pd.Series(df.index).diff().dropna()
        assert (deltas == pd.Timedelta(hours=1)).all()


# ---------------------------------------------------------------------------
# Jena Climate preprocessing logic tests
# ---------------------------------------------------------------------------
class TestJenaPreprocessing:
    """Tests for Jena Climate preprocessing logic."""

    def test_schema_columns(self):
        """Synthetic Jena df has exactly the 5 selected features."""
        df = _make_jena_df()
        expected = {"T (degC)", "p (mbar)", "rh (%)", "wv (m/s)", "wd (deg)"}
        assert set(df.columns) == expected

    def test_no_sentinel_values(self):
        """No -9999.0 sentinel values remain."""
        df = _make_jena_df()
        for col in df.columns:
            assert (df[col] != -9999.0).all(), f"{col} has sentinel values"

    def test_no_nans(self):
        """After preprocessing, no NaN values remain."""
        df = _make_jena_df()
        assert df.isna().sum().sum() == 0

    def test_temperature_range(self):
        """Temperature values are in a plausible range."""
        df = _make_jena_df()
        assert df["T (degC)"].min() > -50
        assert df["T (degC)"].max() < 50


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (synthetic data through prepare_benchmark_data)
# ---------------------------------------------------------------------------
class TestEndToEndPipeline:
    """Tests that synthetic data flows through the full benchmark pipeline."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def _import_prepare(self):
        """Import prepare_benchmark_data from the benchmark script."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark", "scripts/benchmark_standard_dataset.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.prepare_benchmark_data

    @pytest.mark.parametrize("dataset_name,make_fn", [
        ("occupancy", _make_occupancy_df),
        ("power", _make_power_df),
        ("jena", _make_jena_df),
    ])
    def test_pipeline_produces_valid_tensors(self, dataset_name, make_fn, device):
        """Data flows through normalize -> create_sequences -> tensors."""
        prepare = self._import_prepare()
        config = DATASET_CONFIGS[dataset_name]

        df = make_fn()  # module-level helper, no self needed

        data = prepare(
            df,
            lookback=config["lookback"],
            horizon=config["horizon"],
            device=device,
            target_col=config["target_col"],
            split_ratios=config["split_ratios"],
        )

        # Check all expected keys
        for key in ["X_train", "y_train", "X_val", "y_val",
                     "X_test", "y_test", "norm_stats", "target_col"]:
            assert key in data, f"Missing key: {key}"

        # Check shapes
        n_features = df.shape[1]
        assert data["X_train"].ndim == 3
        assert data["X_train"].shape[1] == config["lookback"]
        assert data["X_train"].shape[2] == n_features
        assert data["y_train"].ndim == 2
        assert data["y_train"].shape[1] == config["horizon"]

        # Val and test should have some sequences
        assert data["X_val"].shape[0] > 0, f"No val sequences for {dataset_name}"
        assert data["X_test"].shape[0] > 0, f"No test sequences for {dataset_name}"

        # No NaN in tensors
        assert not torch.isnan(data["X_train"]).any()
        assert not torch.isnan(data["y_train"]).any()

    def test_target_col_stored_in_data(self, device):
        """prepare_benchmark_data stores target_col in returned dict."""
        prepare = self._import_prepare()
        df = _make_occupancy_df(n_rows=200)
        data = prepare(df, lookback=24, horizon=6, device=device,
                       target_col="CO2", split_ratios=[0.7, 0.15, 0.15])
        assert data["target_col"] == "CO2"

    def test_norm_stats_contain_target(self, device):
        """Norm stats dict includes the target column."""
        prepare = self._import_prepare()
        df = _make_jena_df(n_rows=1000)
        data = prepare(df, lookback=168, horizon=24, device=device,
                       target_col="T (degC)", split_ratios=[0.8, 0.1, 0.1])
        assert "T (degC)" in data["norm_stats"]
        mean, std = data["norm_stats"]["T (degC)"]
        assert isinstance(mean, float)
        assert std > 0
