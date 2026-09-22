"""Tests for temporal Von Mises components (src/temporal.py)."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.von_mises_neuron import VonMisesLinear


# ---------------------------------------------------------------------------
# VonMisesLinear phase_shift support
# ---------------------------------------------------------------------------

class TestVonMisesLinearPhaseShift:
    """Tests for the phase_shift parameter added to VonMisesLinear.forward()."""

    def test_no_phase_shift_unchanged(self):
        """Passing phase_shift=None gives identical output to original forward."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(3, 4)
        with torch.no_grad():
            y_default = layer(x)
            y_none = layer(x, phase_shift=None)
        torch.testing.assert_close(y_default, y_none)

    def test_phase_shift_changes_output(self):
        """Non-zero phase_shift produces different output."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(3, 4)
        ps = torch.ones(3, 8, 1) * 0.5
        with torch.no_grad():
            y_orig = layer(x)
            y_shifted = layer(x, phase_shift=ps)
        assert not torch.allclose(y_orig, y_shifted, atol=1e-6)

    def test_phase_shift_periodicity(self):
        """2π phase shift gives same output as no shift (periodicity preserved)."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(3, 4)
        ps_zero = torch.zeros(3, 8, 1)
        ps_2pi = torch.full((3, 8, 1), 2 * math.pi)
        with torch.no_grad():
            y_zero = layer(x, phase_shift=ps_zero)
            y_2pi = layer(x, phase_shift=ps_2pi)
        torch.testing.assert_close(y_zero, y_2pi, atol=1e-5, rtol=1e-5)

    def test_phase_shift_3d_input(self):
        """phase_shift works with 3D input (batch, seq_len, features)."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(2, 10, 4)
        ps = torch.randn(2, 10, 8, 1) * 0.1
        with torch.no_grad():
            y = layer(x, phase_shift=ps)
        assert y.shape == (2, 10, 8)

    def test_phase_shift_gradient_flow(self):
        """Gradients flow through the phase_shift path."""
        layer = VonMisesLinear(4, 8)
        x = torch.randn(3, 4)
        ps = torch.randn(3, 8, 1, requires_grad=True)
        y = layer(x, phase_shift=ps)
        y.sum().backward()
        assert ps.grad is not None
        assert (ps.grad != 0).any()

    def test_input_periodicity_with_phase_shift(self):
        """Input 2π-periodicity preserved even when phase_shift is applied."""
        layer = VonMisesLinear(4, 8)
        layer.eval()
        x = torch.randn(3, 4)
        ps = torch.randn(3, 8, 1)
        with torch.no_grad():
            y1 = layer(x, phase_shift=ps)
            y2 = layer(x + 2 * math.pi, phase_shift=ps)
        torch.testing.assert_close(y1, y2, atol=1e-5, rtol=1e-5)


from src.temporal import extract_angular_time_features


# ---------------------------------------------------------------------------
# Angular time extraction
# ---------------------------------------------------------------------------

class TestExtractAngularTimeFeatures:
    """Tests for extract_angular_time_features()."""

    def test_output_shape(self):
        """Output has shape (T, n_periods)."""
        index = pd.date_range("2021-01-01", periods=48, freq="1h")
        result = extract_angular_time_features(index)
        assert result.shape == (48, 3)

    def test_custom_periods(self):
        """Custom period list controls output columns."""
        index = pd.date_range("2021-01-01", periods=24, freq="1h")
        result = extract_angular_time_features(index, periods=[24])
        assert result.shape == (24, 1)

    def test_values_in_range(self):
        """All output values are in [0, 2π)."""
        index = pd.date_range("2021-01-01", periods=8760, freq="1h")
        result = extract_angular_time_features(index)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_known_hour_angles(self):
        """Hour 0 → 0, hour 6 → π/2, hour 12 → π, hour 18 → 3π/2."""
        index = pd.date_range("2021-01-01", periods=24, freq="1h")
        result = extract_angular_time_features(index, periods=[24])
        expected_0 = 0.0
        expected_6 = 2 * np.pi * 6 / 24
        expected_12 = 2 * np.pi * 12 / 24
        np.testing.assert_allclose(result[0, 0], expected_0, atol=1e-10)
        np.testing.assert_allclose(result[6, 0], expected_6, atol=1e-10)
        np.testing.assert_allclose(result[12, 0], expected_12, atol=1e-10)

    def test_known_dow_angles(self):
        """Weekly angle: continuous hours within the week."""
        # 2021-01-04 is a Monday
        index = pd.date_range("2021-01-04", periods=168, freq="1h")
        result = extract_angular_time_features(index, periods=[168])
        # Monday 00:00 → 0 hours into week → θ = 0
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1e-10)
        # Wednesday 00:00 → 48 hours into week → θ = 2π·48/168
        expected_48 = 2 * np.pi * 48 / 168
        np.testing.assert_allclose(result[48, 0], expected_48, atol=1e-10)
        # Tuesday 12:00 → 36 hours into week → θ = 2π·36/168
        expected_36 = 2 * np.pi * 36 / 168
        np.testing.assert_allclose(result[36, 0], expected_36, atol=1e-10)

    def test_dtype_float64(self):
        """Output is float64 for precision."""
        index = pd.date_range("2021-01-01", periods=24, freq="1h")
        result = extract_angular_time_features(index)
        assert result.dtype == np.float64

    @pytest.mark.parametrize("explicit_origin", [False, True])
    def test_custom_period_is_independent_of_split(self, explicit_origin):
        """A 48-hour phase stays aligned when splits start mid-cycle."""
        index = pd.date_range("2024-02-27 05:30", periods=113, freq="1h")
        kwargs = {"phase_origin": index[0]} if explicit_origin else {}
        whole = extract_angular_time_features(index, [48], **kwargs)
        splits = [index[:53], index[53:81], index[81:]]
        separate = np.concatenate([
            extract_angular_time_features(part, [48], **kwargs) for part in splits
        ])
        np.testing.assert_array_equal(separate, whole)
        steps = np.diff(whole[:, 0]) % (2 * np.pi)
        np.testing.assert_allclose(steps, 2 * np.pi / 48, atol=1e-10)
        if explicit_origin:
            assert whole[0, 0] == 0

    def test_custom_epoch_negative_and_fractional_hours(self):
        """Elapsed cycles wrap before the epoch and resolve fractional hours."""
        index = pd.DatetimeIndex(["1969-12-31 23:30", "1970-01-01 00:30"])
        result = extract_angular_time_features(index, [48])
        np.testing.assert_allclose(result[:, 0], 2 * np.pi * np.array([47.5, 0.5]) / 48)

    def test_calendar_fractional_seconds(self):
        """Daily, weekly, and yearly positions preserve sub-second time."""
        index = pd.DatetimeIndex(["2024-01-01 06:30:15.123456789"])
        hours = 6 + 30 / 60 + 15.123456789 / 3600
        expected = 2 * np.pi * np.array([hours / 24, hours / 168, hours / (366 * 24)])
        np.testing.assert_allclose(extract_angular_time_features(index)[0], expected, atol=1e-14)

    @pytest.mark.parametrize("year,days", [(2023, 365), (2024, 366)])
    def test_annual_year_boundary(self, year, days):
        """The final fractional day approaches 2π and Jan 1 resets to zero."""
        index = pd.DatetimeIndex([
            f"{year}-01-01", f"{year}-12-31 23:30", f"{year + 1}-01-01"
        ])
        result = extract_angular_time_features(index, [8766])[:, 0]
        expected = np.array([0, 2 * np.pi * (1 - 0.5 / (days * 24)), 0])
        np.testing.assert_allclose(result, expected)
        circular_step = (result[2] - result[1]) % (2 * np.pi)
        np.testing.assert_allclose(circular_step, 2 * np.pi * 0.5 / (days * 24))

    def test_leap_day_is_continuous(self):
        """Leap day consumes one ordinary calendar day in the leap year."""
        index = pd.date_range("2024-02-28", periods=3, freq="1D")
        result = extract_angular_time_features(index, [8766])[:, 0]
        np.testing.assert_allclose(np.diff(result), 2 * np.pi / 366)

    def test_week_and_day_boundary(self):
        """Sunday midnight wraps both cycles with the expected circular step."""
        index = pd.DatetimeIndex(["2024-01-07 23:30", "2024-01-08 00:00"])
        result = extract_angular_time_features(index, [24, 168])
        np.testing.assert_allclose(result[1], 0)
        np.testing.assert_allclose((result[1] - result[0]) % (2 * np.pi),
                                   2 * np.pi * 0.5 / np.array([24, 168]))

    @pytest.mark.parametrize("start,local_hours", [
        ("2024-03-31 00:00", [1, 3, 4]),
        ("2024-10-27 00:00", [2, 2, 3]),
    ])
    def test_dst_calendar_and_elapsed_semantics(self, start, local_hours):
        """Calendar cycles follow local clocks; elapsed cycles follow UTC."""
        utc_index = pd.date_range(start, periods=3, freq="1h", tz="UTC")
        local_index = utc_index.tz_convert("Europe/Budapest")
        local = extract_angular_time_features(local_index, [24, 48])
        utc = extract_angular_time_features(utc_index, [24, 48])
        np.testing.assert_allclose(local[:, 0], 2 * np.pi * np.array(local_hours) / 24)
        np.testing.assert_array_equal(local[:, 1], utc[:, 1])
        np.testing.assert_allclose(np.diff(local[:, 1]) % (2 * np.pi), 2 * np.pi / 48)

    @pytest.mark.parametrize("periods", [[], [0], [-48], [np.inf], [np.nan]])
    def test_invalid_periods(self, periods):
        with pytest.raises(ValueError, match="periods"):
            extract_angular_time_features(pd.date_range("2024-01-01", periods=2), periods)

    def test_empty_index(self):
        result = extract_angular_time_features(pd.DatetimeIndex([]), [24, 48, 8766])
        assert result.shape == (0, 3)
        assert result.dtype == np.float64

    def test_rejects_missing_timestamp(self):
        with pytest.raises(ValueError, match="NaT"):
            extract_angular_time_features(pd.DatetimeIndex([pd.NaT]))

    def test_rejects_timezone_mismatch(self):
        with pytest.raises(ValueError, match="timezone awareness"):
            extract_angular_time_features(
                pd.date_range("2024-01-01", periods=2, tz="UTC"),
                [48], phase_origin=pd.Timestamp("2024-01-01"),
            )


from src.temporal import FrequencyBankVonMises


# ---------------------------------------------------------------------------
# FrequencyBankVonMises
# ---------------------------------------------------------------------------

class TestFrequencyBankVonMises:
    """Tests for the frozen-κ frequency bank VM layer."""

    def test_output_shape(self):
        """Forward produces correct output shape."""
        bank = FrequencyBankVonMises(3, 12, freq_bands=[0.5, 2.0, 8.0])
        x = torch.randn(4, 10, 3)
        y = bank(x)
        assert y.shape == (4, 10, 12)

    def test_kappa_frozen(self):
        """κ is not in model.parameters() (stored as buffer)."""
        bank = FrequencyBankVonMises(3, 12, freq_bands=[0.5, 2.0, 8.0])
        param_names = {name for name, _ in bank.named_parameters()}
        assert "kappa" not in param_names
        assert "kappa" in dict(bank.named_buffers())

    def test_kappa_values_correct(self):
        """Each neuron's κ matches its assigned band."""
        bank = FrequencyBankVonMises(3, 9, freq_bands=[0.5, 2.0, 8.0])
        kappa = bank.kappa
        torch.testing.assert_close(kappa[0:3], torch.tensor([0.5, 0.5, 0.5]))
        torch.testing.assert_close(kappa[3:6], torch.tensor([2.0, 2.0, 2.0]))
        torch.testing.assert_close(kappa[6:9], torch.tensor([8.0, 8.0, 8.0]))

    def test_kappa_uneven_allocation(self):
        """Remainder neurons go to first bands."""
        bank = FrequencyBankVonMises(3, 10, freq_bands=[0.5, 2.0, 8.0])
        kappa = bank.kappa
        assert (kappa[:4] == 0.5).all()
        assert (kappa[4:7] == 2.0).all()
        assert (kappa[7:10] == 8.0).all()

    def test_mu_learnable(self):
        """μ has gradients (is learnable)."""
        bank = FrequencyBankVonMises(3, 12, freq_bands=[0.5, 2.0, 8.0])
        x = torch.randn(2, 5, 3)
        y = bank(x)
        y.sum().backward()
        assert bank.mu.grad is not None
        assert (bank.mu.grad != 0).any()

    def test_kappa_no_gradient(self):
        """κ buffer receives no gradient."""
        bank = FrequencyBankVonMises(3, 12, freq_bands=[0.5, 2.0, 8.0])
        x = torch.randn(2, 5, 3)
        y = bank(x)
        y.sum().backward()
        assert bank.kappa.grad is None or not bank.kappa.requires_grad

    def test_single_band(self):
        """Works with a single frequency band."""
        bank = FrequencyBankVonMises(3, 8, freq_bands=[1.0])
        x = torch.randn(2, 5, 3)
        y = bank(x)
        assert y.shape == (2, 5, 8)

    def test_moves_to_device(self):
        """Buffer moves with .to()."""
        bank = FrequencyBankVonMises(3, 6, freq_bands=[0.5, 2.0])
        bank_cpu = bank.to("cpu")
        assert bank_cpu.kappa.device.type == "cpu"


from src.temporal import TemporalVonMisesEncoder


# ---------------------------------------------------------------------------
# TemporalVonMisesEncoder
# ---------------------------------------------------------------------------

class TestTemporalVonMisesEncoder:
    """Tests for the time-conditioned VM encoder with conv bridges."""

    def _make_encoder(self, **kwargs):
        defaults = dict(
            sensor_size=4,
            time_size=3,
            hidden_size=16,
            num_vm_layers=2,
            freq_bands=[0.5, 2.0, 8.0],
            dropout=0.0,
            use_temporal_conv=True,
            conv_kernel_size=3,
        )
        defaults.update(kwargs)
        return TemporalVonMisesEncoder(**defaults)

    def test_output_shape(self):
        """Forward produces (B, T, hidden_size)."""
        enc = self._make_encoder()
        x = torch.randn(4, 20, 7)  # 4 sensor + 3 time
        y = enc(x)
        assert y.shape == (4, 20, 16)

    def test_sensor_time_split(self):
        """Encoder correctly splits sensor and time features."""
        enc = self._make_encoder(sensor_size=4, time_size=3)
        x = torch.zeros(2, 10, 7)
        x[:, :, 4:] = torch.tensor([1.0, 2.0, 3.0])
        y = enc(x)
        assert y.shape == (2, 10, 16)

    def test_gradient_flow_all_params(self):
        """Gradients reach: VM μ, VM κ, freq bank μ, Δμ projections, conv weights."""
        enc = self._make_encoder()
        x = torch.randn(2, 10, 7)
        y = enc(x)
        y.sum().backward()
        has_grad = {}
        for name, param in enc.named_parameters():
            if param.grad is not None and (param.grad != 0).any():
                component = name.split(".")[0]
                has_grad[component] = True
        assert len(has_grad) >= 3, f"Only got gradients in: {list(has_grad.keys())}"

    def test_no_conv_variant(self):
        """use_temporal_conv=False produces valid output."""
        enc = self._make_encoder(use_temporal_conv=False)
        x = torch.randn(2, 10, 7)
        y = enc(x)
        assert y.shape == (2, 10, 16)

    def test_conv_changes_params(self):
        """Conv variant has more parameters than no-conv variant."""
        enc_conv = self._make_encoder(use_temporal_conv=True)
        enc_no_conv = self._make_encoder(use_temporal_conv=False)
        n_conv = sum(p.numel() for p in enc_conv.parameters())
        n_no_conv = sum(p.numel() for p in enc_no_conv.parameters())
        assert n_conv > n_no_conv


from src.temporal import TemporalVonMisesForecaster


# ---------------------------------------------------------------------------
# TemporalVonMisesForecaster
# ---------------------------------------------------------------------------

class TestTemporalVonMisesForecaster:
    """Tests for the complete temporal forecaster model."""

    def _make_model(self, **kwargs):
        defaults = dict(
            sensor_size=4,
            time_periods=[24, 168, 8766],
            hidden_size=16,
            num_vm_layers=2,
            freq_bands=[0.5, 2.0, 8.0],
            dropout=0.0,
            output_size=12,
            use_temporal_conv=True,
            conv_kernel_size=3,
            use_lstm=True,
        )
        defaults.update(kwargs)
        return TemporalVonMisesForecaster(**defaults)

    def test_output_shape(self):
        """Forward produces (B, output_size)."""
        model = self._make_model()
        x = torch.randn(4, 20, 7)  # 4 sensor + 3 time
        y = model(x)
        assert y.shape == (4, 12)

    def test_output_shape_different_horizon(self):
        """Works with different output sizes."""
        model = self._make_model(output_size=24)
        x = torch.randn(2, 30, 7)
        y = model(x)
        assert y.shape == (2, 24)

    def test_backward_pass(self):
        """Full backward pass completes without error."""
        model = self._make_model()
        x = torch.randn(2, 10, 7)
        y = model(x)
        loss = y.sum()
        loss.backward()
        grads_exist = any(
            p.grad is not None and (p.grad != 0).any()
            for p in model.parameters()
        )
        assert grads_exist

    def test_use_lstm_false(self):
        """use_lstm=False uses mean pooling, still produces correct shape."""
        model = self._make_model(use_lstm=False)
        x = torch.randn(2, 10, 7)
        y = model(x)
        assert y.shape == (2, 12)

    def test_no_conv_no_lstm(self):
        """Both ablation flags off still works."""
        model = self._make_model(use_temporal_conv=False, use_lstm=False)
        x = torch.randn(2, 10, 7)
        y = model(x)
        assert y.shape == (2, 12)

    def test_single_freq_band(self):
        """Works with a single frequency band."""
        model = self._make_model(freq_bands=[2.0])
        x = torch.randn(2, 10, 7)
        y = model(x)
        assert y.shape == (2, 12)

    def test_different_time_periods(self):
        """Custom time periods list changes input expectation."""
        model = self._make_model(time_periods=[24])
        x = torch.randn(2, 10, 5)  # 4 sensor + 1 time
        y = model(x)
        assert y.shape == (2, 12)

    def test_get_vm_layers(self):
        """Interpretability accessor returns VM layers."""
        model = self._make_model(num_vm_layers=3)
        vm_layers = model.get_vm_layers()
        assert len(vm_layers) == 3
        assert all(isinstance(layer, VonMisesLinear) for layer in vm_layers)

    def test_get_freq_bank(self):
        """Can access frequency bank for interpretability."""
        model = self._make_model()
        bank = model.get_freq_bank()
        assert isinstance(bank, FrequencyBankVonMises)


from src.preprocessing import normalize
from src.split import chronological_split
from src.sequences import create_sequences


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestTemporalPipeline:
    """Tests for angular features surviving the data pipeline."""

    def test_angular_features_bypass_normalization(self):
        """Angular features stay in [0, 2π) when added after normalize()."""
        # Create synthetic DataFrame with DatetimeIndex
        index = pd.date_range("2021-01-01", periods=200, freq="1h")
        df = pd.DataFrame({
            "Temperature": np.random.randn(200) * 3 + 22,
            "CO2": np.random.randn(200) * 50 + 600,
        }, index=index)

        # Pipeline: split → normalize → extract angles → concat → sequence
        train_df, val_df, test_df = chronological_split(df, ratios=[0.8, 0.1, 0.1])

        train_norm, stats = normalize(train_df)

        # Extract angular features from original (pre-norm) index
        angles = extract_angular_time_features(train_norm.index, periods=[24, 168])

        # Concat: angular features are appended to normalized sensor features
        for i, p in enumerate([24, 168]):
            train_norm[f"angle_{p}"] = angles[:, i]

        # Verify angular columns are still in [0, 2π)
        for p in [24, 168]:
            col = train_norm[f"angle_{p}"]
            assert col.min() >= 0, f"angle_{p} has values < 0"
            assert col.max() < 2 * np.pi, f"angle_{p} has values >= 2π"

        # Verify sensor columns are z-scored (mean ≈ 0, std ≈ 1)
        assert abs(train_norm["Temperature"].mean()) < 0.5
        assert abs(train_norm["CO2"].mean()) < 0.5

    def test_full_pipeline_to_model(self):
        """End-to-end: DataFrame → sequences → model forward pass."""
        index = pd.date_range("2021-01-01", periods=200, freq="1h")
        df = pd.DataFrame({
            "Temperature": np.random.randn(200) * 3 + 22,
            "CO2": np.random.randn(200) * 50 + 600,
        }, index=index)

        train_df, _, _ = chronological_split(df, ratios=[0.8, 0.1, 0.1])
        train_norm, _ = normalize(train_df)

        angles = extract_angular_time_features(train_norm.index, periods=[24, 168])
        for i, p in enumerate([24, 168]):
            train_norm[f"angle_{p}"] = angles[:, i]

        X, y, _ = create_sequences(
            train_norm, lookback=24, horizon=6, target_col="CO2"
        )
        X_tensor = torch.tensor(X, dtype=torch.float32)

        model = TemporalVonMisesForecaster(
            sensor_size=2,  # Temperature, CO2
            time_periods=[24, 168],
            hidden_size=8,
            num_vm_layers=1,
            output_size=6,
            use_temporal_conv=False,
            use_lstm=False,
        )
        with torch.no_grad():
            pred = model(X_tensor)
        assert pred.shape == (X_tensor.shape[0], 6)
