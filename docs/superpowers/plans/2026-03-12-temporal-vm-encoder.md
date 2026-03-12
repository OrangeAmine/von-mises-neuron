# Temporal Von Mises Encoder Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add time-aware temporal features to the Von Mises neuron forecaster via angular time extraction, a frozen-κ frequency bank, time-conditioned phase fusion, and depthwise temporal convolutions.

**Architecture:** Angular time features (hour, day-of-week, season as raw angles) feed through a FrequencyBankVonMises layer with frozen κ bands. Its output produces per-neuron phase shifts (Δμ) that modulate the sensor VM encoder. Depthwise Conv1d bridges between VM layers add local temporal mixing. An LSTM backbone handles global sequence dynamics.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-temporal-vm-encoder-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/temporal.py` | Create | `extract_angular_time_features()`, `FrequencyBankVonMises`, `TemporalVonMisesEncoder`, `TemporalVonMisesForecaster` |
| `src/von_mises_neuron.py` | Modify (lines 120-152) | Add `phase_shift` param to `VonMisesLinear.forward()` |
| `tests/test_temporal.py` | Create | All tests for new components |
| `scripts/benchmark_standard_dataset.py` | Modify | Add `prepare_temporal_data()`, `benchmark_temporal_von_mises()`, wire into `main()` |

---

## Chunk 1: Foundation — phase_shift support + angular time extraction

### Task 1: Add `phase_shift` parameter to `VonMisesLinear.forward()`

**Files:**
- Modify: `src/von_mises_neuron.py:120-152`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for phase_shift**

Create `tests/test_temporal.py`:

```python
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
        # shape: (batch, out_features, 1) for 2D input
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
        # shape: (batch, seq_len, out_features, 1)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal.py::TestVonMisesLinearPhaseShift -v`
Expected: FAIL — `forward()` does not accept `phase_shift`

- [ ] **Step 3: Implement phase_shift in VonMisesLinear.forward()**

In `src/von_mises_neuron.py`, modify the `forward` method (lines 120-152). Change the signature and the `cos_term` computation:

```python
def forward(self, x: Tensor, phase_shift: Optional[Tensor] = None) -> Tensor:
    """Apply Von Mises activation.

    Args:
        x: Input tensor, shape (..., in_features).
        phase_shift: Optional per-neuron phase correction, shape
            (..., out_features, 1). Broadcasts across in_features.
            Used by TemporalVonMisesEncoder to condition phases on time.

    Returns:
        Output tensor, shape (..., out_features).
    """
    # Eq. κ > 0 via softplus: κ = log(1 + exp(raw))
    kappa = nn.functional.softplus(self.log_kappa)  # (out, in)

    # Compute effective phase: base μ + optional time-conditioned shift
    # phase_shift (..., out, 1) broadcasts across in_features dimension
    effective_mu = self.mu if phase_shift is None else self.mu + phase_shift

    # Broadcast x over the output dimension:
    # x: (..., in) → (..., 1, in)  |  effective_mu: (out, in) → broadcast
    cos_term = torch.cos(x.unsqueeze(-2) + effective_mu)  # (..., out, in)

    # z_j = Σᵢ κⱼᵢ · cos(xᵢ + μⱼᵢ + Δμⱼ)  —  the log-space VM response
    z = (kappa * cos_term).sum(dim=-1)  # (..., out)

    if self.log_output:
        out = z
    else:
        out = torch.exp(z)

    if self.bias is not None:
        out = out + self.bias

    return out
```

Also add `Optional` to the imports if not already present (it is — line 3 imports it from `typing`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestVonMisesLinearPhaseShift -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Run existing tests to verify backward compatibility**

Run: `pytest tests/test_von_mises_neuron.py -v`
Expected: All existing tests PASS (no change in default behavior)

- [ ] **Step 6: Commit**

```bash
git add src/von_mises_neuron.py tests/test_temporal.py
git commit -m "feat: add optional phase_shift parameter to VonMisesLinear.forward()"
```

---

### Task 2: Implement `extract_angular_time_features()`

**Files:**
- Create: `src/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for angular time extraction**

Append to `tests/test_temporal.py`:

```python
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
        # Default 3 periods: daily, weekly, annual
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
        expected_6 = 2 * np.pi * 6 / 24  # π/2
        expected_12 = 2 * np.pi * 12 / 24  # π
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
        # This tests mid-day interpolation (distinguishes continuous from discrete)
        expected_36 = 2 * np.pi * 36 / 168
        np.testing.assert_allclose(result[36, 0], expected_36, atol=1e-10)

    def test_dtype_float64(self):
        """Output is float64 for precision."""
        index = pd.date_range("2021-01-01", periods=24, freq="1h")
        result = extract_angular_time_features(index)
        assert result.dtype == np.float64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal.py::TestExtractAngularTimeFeatures -v`
Expected: FAIL — `ImportError: cannot import name 'extract_angular_time_features' from 'src.temporal'`

- [ ] **Step 3: Implement extract_angular_time_features()**

Create `src/temporal.py`:

```python
"""Temporal Von Mises components: time-aware encoding and forecasting.

Extends the Von Mises neuron with explicit temporal features:
  1. Angular time extraction: timestamps → raw angles in [0, 2π)
  2. Frequency bank: frozen-κ multi-scale VM layer
  3. Phase fusion: time-conditioned Δμ shifts for sensor VM neurons
  4. Temporal conv bridges: depthwise Conv1d for local temporal mixing

See docs/superpowers/specs/2026-03-12-temporal-vm-encoder-design.md
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.von_mises_neuron import VonMisesLinear


def extract_angular_time_features(
    index: pd.DatetimeIndex,
    periods: Optional[List[int]] = None,
) -> np.ndarray:
    """Convert a DatetimeIndex into angular time features in [0, 2π).

    Each period maps a cyclic time component to an angle using a unified
    formula: θ = 2π · (fractional_position_within_period).

      - 24   → θ = 2π · hour / 24               (daily cycle)
      - 168  → θ = 2π · (dow·24 + hour) / 168    (weekly cycle, continuous)
      - 8766 → θ = 2π · day_of_year / 365.25     (annual cycle)

    The weekly formula uses continuous hours within the week (not discrete
    dayofweek) so the angle interpolates smoothly between days. This is
    consistent with how the generic-period formula works.

    These raw angles are designed to be consumed directly by Von Mises
    neurons, which learn phase shifts μ to align with periodic patterns.

    Args:
        index: DatetimeIndex from a sensor DataFrame.
        periods: List of periods in hours. Default [24, 168, 8766].

    Returns:
        np.ndarray of shape (T, len(periods)), dtype float64, values in [0, 2π).
    """
    if periods is None:
        periods = [24, 168, 8766]

    # Fractional hour within the day (0.0 to ~23.99)
    hours = index.hour + index.minute / 60.0 + index.second / 3600.0

    features = []
    for p in periods:
        if p == 24:
            # Daily: fractional hour → angle
            theta = 2.0 * np.pi * hours / 24.0
        elif p == 168:
            # Weekly: continuous hours within the week → angle
            # dayofweek * 24 + fractional_hour gives position in [0, 168)
            weekly_hours = index.dayofweek * 24.0 + hours
            theta = 2.0 * np.pi * weekly_hours / 168.0
        elif p == 8766:
            # Annual: day-of-year → angle (day_of_year starts at 1)
            theta = 2.0 * np.pi * index.dayofyear / 365.25
        else:
            # Generic: compute hours since start, mod by period
            t_hours = (index - index[0]).total_seconds() / 3600.0
            theta = 2.0 * np.pi * (t_hours % p) / p

        # Wrap to [0, 2π) for safety
        theta = np.asarray(theta, dtype=np.float64) % (2.0 * np.pi)
        features.append(theta)

    return np.column_stack(features)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestExtractAngularTimeFeatures -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/temporal.py tests/test_temporal.py
git commit -m "feat: add extract_angular_time_features() for angular time encoding"
```

---

## Chunk 2: FrequencyBankVonMises + TemporalVonMisesEncoder

### Task 3: Implement `FrequencyBankVonMises`

**Files:**
- Modify: `src/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for FrequencyBankVonMises**

Append to `tests/test_temporal.py`:

```python
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
        # But it should be in state_dict (as buffer)
        assert "kappa" in dict(bank.named_buffers())

    def test_kappa_values_correct(self):
        """Each neuron's κ matches its assigned band."""
        bank = FrequencyBankVonMises(3, 9, freq_bands=[0.5, 2.0, 8.0])
        kappa = bank.kappa  # (9,) buffer
        # 9 neurons / 3 bands = 3 per band
        torch.testing.assert_close(kappa[0:3], torch.tensor([0.5, 0.5, 0.5]))
        torch.testing.assert_close(kappa[3:6], torch.tensor([2.0, 2.0, 2.0]))
        torch.testing.assert_close(kappa[6:9], torch.tensor([8.0, 8.0, 8.0]))

    def test_kappa_uneven_allocation(self):
        """Remainder neurons go to first bands."""
        bank = FrequencyBankVonMises(3, 10, freq_bands=[0.5, 2.0, 8.0])
        kappa = bank.kappa  # (10,)
        # 10 / 3 = 3 remainder 1 → first band gets 4, others get 3
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
        """Buffer moves with .to() (important for GPU support)."""
        bank = FrequencyBankVonMises(3, 6, freq_bands=[0.5, 2.0])
        bank_cpu = bank.to("cpu")
        assert bank_cpu.kappa.device.type == "cpu"
        # Just verify .to() doesn't crash; GPU test only if available
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal.py::TestFrequencyBankVonMises -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement FrequencyBankVonMises**

Append to `src/temporal.py`:

```python
class FrequencyBankVonMises(nn.Module):
    """Von Mises layer with frozen κ organized into frequency bands.

    Neurons are divided equally among bands. Each band has a frozen
    concentration κ (stored as a buffer) that controls its temporal
    resolution: high κ → sharp/fast, low κ → broad/slow. Only the
    phase shifts μ are learned.

    This forces multi-scale decomposition: the network *must* use all
    timescales rather than collapsing to a single dominant one.

    Eq:  z_j = κ_band(j) · cos(θᵢ + μⱼᵢ)   (summed over i)

    Args:
        in_features: Number of angular input features (= len(time_periods)).
        out_features: Total number of neurons across all bands.
        freq_bands: List of κ values, one per band. Default [0.5, 2.0, 8.0].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        freq_bands: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if freq_bands is None:
            freq_bands = [0.5, 2.0, 8.0]

        self.in_features = in_features
        self.out_features = out_features
        self.freq_bands = freq_bands
        n_bands = len(freq_bands)

        # Allocate neurons to bands: equal split, remainder to first bands
        base_count = out_features // n_bands
        remainder = out_features % n_bands
        band_sizes = [base_count + (1 if i < remainder else 0) for i in range(n_bands)]

        # Build frozen κ buffer: (out_features,)
        kappa_values = []
        for band_kappa, size in zip(freq_bands, band_sizes):
            kappa_values.extend([band_kappa] * size)
        self.register_buffer("kappa", torch.tensor(kappa_values, dtype=torch.float32))

        # Learnable μ: phase shifts, (out_features, in_features)
        self.mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.uniform_(self.mu, 0, 2 * math.pi)

        # Optional bias
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the frequency bank.

        Args:
            x: Angular input, shape (..., in_features).

        Returns:
            Log-space VM response, shape (..., out_features).
        """
        # cos(θᵢ + μⱼᵢ) for all (neuron j, input i) pairs
        # x: (..., in) → (..., 1, in)  |  mu: (out, in)
        cos_term = torch.cos(x.unsqueeze(-2) + self.mu)  # (..., out, in)

        # κ: (out,) → (out, 1) for broadcasting over in_features
        kappa_expanded = self.kappa.unsqueeze(-1)  # (out, 1)

        # z_j = Σᵢ κ_band(j) · cos(θᵢ + μⱼᵢ)
        z = (kappa_expanded * cos_term).sum(dim=-1)  # (..., out)

        # No exp() here — the Δμ projection expects unbounded reals, not positive values
        return z + self.bias

    def get_band_indices(self) -> List[List[int]]:
        """Return neuron indices for each frequency band.

        Useful for interpretability: inspect phases per band.
        """
        indices = []
        offset = 0
        n_bands = len(self.freq_bands)
        base_count = self.out_features // n_bands
        remainder = self.out_features % n_bands
        for i in range(n_bands):
            size = base_count + (1 if i < remainder else 0)
            indices.append(list(range(offset, offset + size)))
            offset += size
        return indices
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestFrequencyBankVonMises -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/temporal.py tests/test_temporal.py
git commit -m "feat: add FrequencyBankVonMises with frozen-κ multi-scale bands"
```

---

### Task 4: Implement `TemporalVonMisesEncoder`

**Files:**
- Modify: `src/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for TemporalVonMisesEncoder**

Append to `tests/test_temporal.py`:

```python
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
        # All zeros for sensor, known angles for time
        x = torch.zeros(2, 10, 7)
        x[:, :, 4:] = torch.tensor([1.0, 2.0, 3.0])  # time features
        y = enc(x)
        assert y.shape == (2, 10, 16)

    def test_gradient_flow_all_params(self):
        """Gradients reach: VM μ, VM κ, freq bank μ, Δμ projections, conv weights."""
        enc = self._make_encoder()
        x = torch.randn(2, 10, 7)
        y = enc(x)
        y.sum().backward()
        # Check at least one gradient is non-zero in each component
        has_grad = {}
        for name, param in enc.named_parameters():
            if param.grad is not None and (param.grad != 0).any():
                # Group by component
                component = name.split(".")[0]
                has_grad[component] = True
        # Should have gradients in freq_bank, delta_mu_projections, vm_layers
        assert len(has_grad) >= 3, f"Only got gradients in: {list(has_grad.keys())}"

    def test_no_conv_variant(self):
        """use_temporal_conv=False produces valid output (no conv, no residual)."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal.py::TestTemporalVonMisesEncoder -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement TemporalVonMisesEncoder**

Append to `src/temporal.py`:

```python
class TemporalVonMisesEncoder(nn.Module):
    """Time-conditioned Von Mises encoder with optional temporal conv bridges.

    Architecture per layer::

        FrequencyBank(time_features)
          → Linear → Δμ (per-neuron phase shift)
        VonMisesLinear(sensor_features, phase_shift=Δμ)
          → LayerNorm
          → [DepthwiseConv1d → Residual]  (if use_temporal_conv)
          → Dropout

    The frequency bank runs once on angular time features and produces
    phase corrections for each VM layer via separate linear projections.

    Args:
        sensor_size: Number of sensor input features.
        time_size: Number of angular time features (= len(time_periods)).
        hidden_size: Working dimension for VM layers.
        num_vm_layers: Number of stacked VM layers.
        freq_bands: κ values for frequency bank bands.
        dropout: Dropout rate.
        use_temporal_conv: If True, add depthwise Conv1d + residual per layer.
        conv_kernel_size: Kernel size for depthwise Conv1d.
    """

    def __init__(
        self,
        sensor_size: int,
        time_size: int,
        hidden_size: int = 64,
        num_vm_layers: int = 2,
        freq_bands: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_temporal_conv: bool = True,
        conv_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.sensor_size = sensor_size
        self.time_size = time_size
        self.hidden_size = hidden_size
        self.num_vm_layers = num_vm_layers
        self._use_temporal_conv = use_temporal_conv

        if freq_bands is None:
            freq_bands = [0.5, 2.0, 8.0]

        # Frequency bank: processes angular time features once
        # Output dimension = hidden_size (same as sensor pathway)
        self.freq_bank = FrequencyBankVonMises(
            in_features=time_size,
            out_features=hidden_size,
            freq_bands=freq_bands,
        )

        # Per-layer Δμ projections: map freq_bank output to phase shifts
        self.delta_mu_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_vm_layers)
        ])

        # VM layers for sensor features
        self.vm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_dim = sensor_size
        for _ in range(num_vm_layers):
            self.vm_layers.append(
                VonMisesLinear(
                    in_dim, hidden_size,
                    bias=True,
                    log_output=True,  # log-space for safe inter-layer stacking
                    init_kappa=1.0,
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size

        # Optional temporal conv bridges
        if use_temporal_conv:
            self.convs = nn.ModuleList([
                nn.Conv1d(
                    hidden_size, hidden_size,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    groups=hidden_size,  # depthwise: each channel independent
                )
                for _ in range(num_vm_layers)
            ])
            # First-layer residual projection (sensor_size → hidden_size)
            if sensor_size != hidden_size:
                self.residual_proj = nn.Linear(sensor_size, hidden_size)
            else:
                self.residual_proj = None
        else:
            self.convs = None
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, sensor_size + time_size).
               Sensor columns first, angular time columns last.

        Returns:
            Encoded tensor, shape (B, T, hidden_size).
        """
        # Split sensor and time features
        x_sensor = x[..., :self.sensor_size]   # (B, T, sensor_size)
        x_time = x[..., self.sensor_size:]     # (B, T, time_size)

        # Run frequency bank once on angular time features
        # freq_out: (B, T, hidden_size) — log-space VM response
        freq_out = self.freq_bank(x_time)

        # Process sensor features through VM layers with time-conditioned phases
        h = x_sensor
        for i in range(self.num_vm_layers):
            residual = h

            # Compute time-conditioned phase shift for this layer
            # (B, T, hidden_size) → (B, T, hidden_size) → unsqueeze → (B, T, hidden_size, 1)
            delta_mu = self.delta_mu_projections[i](freq_out).unsqueeze(-1)

            # VM layer with phase conditioning
            h = self.vm_layers[i](h, phase_shift=delta_mu)
            h = self.layer_norms[i](h)

            # Optional temporal conv + residual
            if self._use_temporal_conv and self.convs is not None:
                # Conv1d expects (B, C, T)
                h_conv = h.transpose(1, 2)           # (B, H, T)
                h_conv = F.gelu(self.convs[i](h_conv))
                h = h_conv.transpose(1, 2)           # (B, T, H)

                # Residual: project first layer if dimensions mismatch
                if i == 0 and self.residual_proj is not None:
                    residual = self.residual_proj(residual)
                h = h + residual

            h = self.dropout(h)

        return h
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestTemporalVonMisesEncoder -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/temporal.py tests/test_temporal.py
git commit -m "feat: add TemporalVonMisesEncoder with phase fusion and conv bridges"
```

---

## Chunk 3: TemporalVonMisesForecaster + benchmark integration

### Task 5: Implement `TemporalVonMisesForecaster`

**Files:**
- Modify: `src/temporal.py`
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write failing tests for TemporalVonMisesForecaster**

Append to `tests/test_temporal.py`:

```python
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
        # Verify at least some gradients exist
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_temporal.py::TestTemporalVonMisesForecaster -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement TemporalVonMisesForecaster**

Append to `src/temporal.py`:

```python
class TemporalVonMisesForecaster(nn.Module):
    """Time-aware Von Mises forecaster with multi-scale periodic encoding.

    Architecture::

        Input (B, T, F_sensor + F_time)
          → TemporalVonMisesEncoder
              [FrequencyBank + Phase Fusion + Conv Bridges] × N
          → LSTM backbone (or mean pool)
          → Linear output head

    The model expects angular time features (from extract_angular_time_features)
    concatenated after sensor features. sensor_size tells the model where to split.

    Args:
        sensor_size: Number of raw sensor features (before time columns).
        time_periods: Periods in hours for angular features (determines time_size).
        hidden_size: Working dimension.
        num_vm_layers: Number of stacked VM layers in the encoder.
        freq_bands: κ values for frequency bank bands.
        dropout: Dropout rate.
        output_size: Forecast horizon.
        use_temporal_conv: If True, add depthwise Conv1d bridges.
        conv_kernel_size: Kernel size for temporal convolutions.
        use_lstm: If True, use LSTM backbone. If False, mean pool over time.
    """

    def __init__(
        self,
        sensor_size: int,
        time_periods: Optional[List[int]] = None,
        hidden_size: int = 64,
        num_vm_layers: int = 2,
        freq_bands: Optional[List[float]] = None,
        dropout: float = 0.0,
        output_size: int = 1,
        use_temporal_conv: bool = True,
        conv_kernel_size: int = 3,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()
        if time_periods is None:
            time_periods = [24, 168, 8766]
        if freq_bands is None:
            freq_bands = [0.5, 2.0, 8.0]

        self.sensor_size = sensor_size
        self.time_periods = time_periods
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._use_lstm = use_lstm

        time_size = len(time_periods)

        # --- Temporal VM encoder ---
        self.encoder = TemporalVonMisesEncoder(
            sensor_size=sensor_size,
            time_size=time_size,
            hidden_size=hidden_size,
            num_vm_layers=num_vm_layers,
            freq_bands=freq_bands,
            dropout=dropout,
            use_temporal_conv=use_temporal_conv,
            conv_kernel_size=conv_kernel_size,
        )

        # --- Temporal backbone ---
        if use_lstm:
            self.lstm: Optional[nn.LSTM] = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
        else:
            self.lstm = None

        # --- Output head ---
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, T, sensor_size + len(time_periods)).

        Returns:
            Forecast tensor, shape (B, output_size).
        """
        # Temporal VM encode: (B, T, F) → (B, T, H)
        h = self.encoder(x)

        if self.lstm is not None:
            # LSTM: capture global temporal dependencies
            _, (h_n, _) = self.lstm(h)
            h_final = h_n.squeeze(0)  # (B, H)
        else:
            # Mean pool over time
            h_final = h.mean(dim=1)  # (B, H)

        return self.output_head(h_final)

    def get_vm_layers(self) -> List[VonMisesLinear]:
        """Return VM layers from the encoder for interpretability."""
        return list(self.encoder.vm_layers)

    def get_freq_bank(self) -> FrequencyBankVonMises:
        """Return the frequency bank for interpretability."""
        return self.encoder.freq_bank
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestTemporalVonMisesForecaster -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run all temporal tests together**

Run: `pytest tests/test_temporal.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/temporal.py tests/test_temporal.py
git commit -m "feat: add TemporalVonMisesForecaster with LSTM backbone and ablation flags"
```

---

### Task 6: Add pipeline test for normalization bypass

**Files:**
- Test: `tests/test_temporal.py`

- [ ] **Step 1: Write pipeline integration test**

Append to `tests/test_temporal.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_temporal.py::TestTemporalPipeline -v`
Expected: All 2 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_temporal.py
git commit -m "test: add pipeline integration tests for angular feature normalization bypass"
```

---

### Task 7: Add benchmark integration

**Files:**
- Modify: `scripts/benchmark_standard_dataset.py`

- [ ] **Step 1: Add imports and `prepare_temporal_data()` to benchmark script**

At the top of `scripts/benchmark_standard_dataset.py`, after the existing imports from `src.*`, add:

```python
from src.temporal import (
    extract_angular_time_features,
    TemporalVonMisesForecaster,
)
```

Add a new function `prepare_temporal_data()` after the existing `prepare_benchmark_data()` function (after line 242):

```python
def prepare_temporal_data(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    device: torch.device,
    time_periods: List[int] = None,
    target_col: str = "CO2",
    split_ratios: List[float] = None,
) -> Dict[str, Any]:
    """Split, normalize, add angular time features, and window data.

    Key difference from prepare_benchmark_data: angular time features are
    added AFTER normalization to preserve their [0, 2π) range. Sensor
    features are z-scored normally.

    Args:
        df: DataFrame with sensor columns and DatetimeIndex.
        lookback: Number of past timesteps per input window.
        horizon: Number of future timesteps to predict.
        device: PyTorch device.
        time_periods: Periods in hours for angular features.
        target_col: Column name for the prediction target.
        split_ratios: Train/val/test ratios.

    Returns:
        Dict with X/y tensors, norm_stats, config.
    """
    if time_periods is None:
        time_periods = [24, 168]
    if split_ratios is None:
        split_ratios = [0.8, 0.1, 0.1]

    # Capture sensor column count BEFORE augmentation — do not move this
    sensor_size = len(df.columns)

    train_df, val_df, test_df = chronological_split(df, ratios=split_ratios)

    # Normalize sensor features only
    train_norm, stats = normalize(train_df, stats=None)
    val_norm, _ = normalize(val_df, stats=stats)
    test_norm, _ = normalize(test_df, stats=stats)

    # Add angular time features AFTER normalization (bypass z-scoring).
    # Use .copy() to avoid mutating the normalized DataFrames in place.
    augmented = []
    for split_norm in [train_norm, val_norm, test_norm]:
        aug = split_norm.copy()
        angles = extract_angular_time_features(aug.index, periods=time_periods)
        for i, p in enumerate(time_periods):
            aug[f"angle_{p}"] = angles[:, i]
        augmented.append(aug)
    train_aug, val_aug, test_aug = augmented

    X_train_np, y_train_np, _ = create_sequences(
        train_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_val_np, y_val_np, _ = create_sequences(
        val_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )
    X_test_np, y_test_np, _ = create_sequences(
        test_aug, lookback=lookback, horizon=horizon, target_col=target_col
    )

    data = {
        "X_train": torch.tensor(X_train_np, dtype=torch.float32).to(device),
        "y_train": torch.tensor(y_train_np, dtype=torch.float32).to(device),
        "X_val": torch.tensor(X_val_np, dtype=torch.float32).to(device),
        "y_val": torch.tensor(y_val_np, dtype=torch.float32).to(device),
        "X_test": torch.tensor(X_test_np, dtype=torch.float32).to(device),
        "y_test": torch.tensor(y_test_np, dtype=torch.float32).to(device),
        "norm_stats": stats,
        "horizon": horizon,
        "lookback": lookback,
        "target_col": target_col,
        "sensor_size": sensor_size,
        "time_periods": time_periods,
    }

    return data
```

- [ ] **Step 2: Add `benchmark_temporal_von_mises()` function**

Add after the existing `benchmark_handcrafted()` function (after line 480):

```python
def benchmark_temporal_von_mises(
    data: Dict[str, Any], epochs: int, device: torch.device
) -> Dict[str, float]:
    """Train and evaluate the Temporal Von Mises forecaster."""
    horizon = data["horizon"]
    sensor_size = data["sensor_size"]
    time_periods = data["time_periods"]

    model = TemporalVonMisesForecaster(
        sensor_size=sensor_size,
        time_periods=time_periods,
        hidden_size=64,
        num_vm_layers=2,
        freq_bands=[0.5, 2.0, 8.0],
        dropout=0.1,
        output_size=horizon,
        use_temporal_conv=True,
        conv_kernel_size=3,
    ).to(device)

    model = _train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        max_epochs=epochs,
    )

    with torch.no_grad():
        y_pred = model(data["X_test"])

    return compute_benchmark_metrics(y_pred, data["y_test"], data["norm_stats"])
```

- [ ] **Step 3: Wire into `main()`**

In `main()`, after the data pipeline preparation section (around line 696), add the temporal data preparation:

```python
    # Temporal pipeline for TemporalVonMisesForecaster (angular time features)
    data_temporal = prepare_temporal_data(
        df, args.lookback, args.horizon, device,
        time_periods=[24, 168],
    )
    print(f"  Temporal features: {data_temporal['X_train'].shape[2]} "
          f"(+{data_temporal['X_train'].shape[2] - data['X_train'].shape[2]} angular)")
```

Change the `n_phases` computation from `7 if run_baselines else 4` to `8 if run_baselines else 5` to account for the new TemporalVM phase.

After the existing VM benchmark section (around line 713, after `vm_passed` is computed), add a new numbered phase:

```python
    # --- Train & evaluate TemporalVonMisesForecaster ---
    phase_tvm = 4 if not run_baselines else 4
    print(f"\n[{phase_tvm}/{n_phases}] Training TemporalVonMisesForecaster...")
    seed_everything(42)
    tvm_result = _run_and_report(
        "TemporalVM", benchmark_temporal_von_mises,
        data_temporal, args.epochs, device
    )
    tvm_model_tmp = TemporalVonMisesForecaster(
        sensor_size=data_temporal["sensor_size"],
        time_periods=data_temporal["time_periods"],
        hidden_size=64, num_vm_layers=2,
        freq_bands=[0.5, 2.0, 8.0], dropout=0.1,
        output_size=data["horizon"],
    )
    tvm_result["params"] = _count_params(tvm_model_tmp)
    del tvm_model_tmp
    all_results["TemporalVM"] = tvm_result
```

The `n_phases` was already updated above. The baseline phases (LSTM=5, Transformer=6, HandCrafted=7) shift by +1 to accommodate TemporalVM at phase 4. Update those phase numbers in their print statements accordingly. The final "Generating report" phase becomes phase 8 (baselines) or 5 (skip-baselines).

- [ ] **Step 4: Run the benchmark to verify integration**

Run: `python scripts/benchmark_standard_dataset.py --epochs 50 --skip-baselines`
Expected: Both VonMises and TemporalVM results appear. TemporalVM meets acceptance thresholds (RMSE<35, MAE<25, R²>0.20, MAPE<5%). Exit code 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark_standard_dataset.py
git commit -m "feat: integrate TemporalVonMisesForecaster into benchmark comparison"
```

---

### Task 8: Add success criteria validation to benchmark

**Files:**
- Modify: `scripts/benchmark_standard_dataset.py`

The spec requires two comparative checks beyond the absolute thresholds:
- **SC3:** TemporalVM must not regress more than 10% on any metric vs. base VonMises
- **SC4:** TemporalVM with `use_lstm=False` must outperform base VonMises with `use_lstm=False`

- [ ] **Step 1: Add `check_regression()` function**

Add after `check_thresholds()` in the benchmark script:

```python
def check_regression(
    new_metrics: Dict[str, float],
    base_metrics: Dict[str, float],
    max_regression_pct: float = 10.0,
) -> Tuple[bool, List[str]]:
    """Check that new model does not regress vs. base beyond a threshold.

    For RMSE, MAE, MAPE: new must not exceed base by more than max_regression_pct.
    For R2: new must not be below base by more than max_regression_pct (absolute).
    """
    failures = []
    for metric in ["rmse", "mae", "mape"]:
        base_val = base_metrics[metric]
        new_val = new_metrics[metric]
        if base_val > 0:
            pct_change = (new_val - base_val) / base_val * 100
            if pct_change > max_regression_pct:
                failures.append(
                    f"  {metric.upper()}: {new_val:.2f} is {pct_change:.1f}% worse "
                    f"than base {base_val:.2f} (max allowed: {max_regression_pct}%)"
                )
    # R2: higher is better, check absolute difference
    r2_diff = base_metrics["r2"] - new_metrics["r2"]
    if r2_diff > max_regression_pct / 100:
        failures.append(
            f"  R2: {new_metrics['r2']:.4f} is {r2_diff:.4f} below "
            f"base {base_metrics['r2']:.4f}"
        )
    return len(failures) == 0, failures
```

- [ ] **Step 2: Add regression check in `main()` after TemporalVM benchmark**

After the TemporalVM results are collected, add:

```python
    # Check SC3: TemporalVM must not regress >10% vs. base VonMises
    tvm_ok, tvm_failures = check_regression(
        tvm_result["metrics"], vm_result["metrics"]
    )
    if not tvm_ok:
        print("\n  WARNING: TemporalVM regressed vs. base VonMises:")
        for f in tvm_failures:
            print(f"    {f}")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_standard_dataset.py
git commit -m "feat: add regression check for TemporalVM vs. base VonMises (spec SC3)"
```

---

### Task 9: Run full test suite and benchmark

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS (including existing tests — no regressions)

- [ ] **Step 2: Run full benchmark with baselines**

Run: `python scripts/benchmark_standard_dataset.py --epochs 50`
Expected: All models train and evaluate. TemporalVM appears in comparison table. VM thresholds still pass. Regression check passes. Exit code 0.

- [ ] **Step 3: Final commit if any adjustments were needed**

```bash
git add -A
git commit -m "chore: final adjustments after full test suite and benchmark run"
```
