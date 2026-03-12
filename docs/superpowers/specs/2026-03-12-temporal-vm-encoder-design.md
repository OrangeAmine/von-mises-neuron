# Temporal Von Mises Encoder — Design Spec

**Date:** 2026-03-12
**Status:** Approved
**Priority stack:** A (explicit time) → C (multi-scale) → B (reduce LSTM dependency)

## Problem

The current `VonMisesForecaster` processes each timestep independently through its VM encoder. It has no explicit knowledge of *when* a timestep occurs — no hour-of-day, day-of-week, or seasonal signal. Ablation studies show the LSTM backbone carries ~99% of temporal learning; the VM encoder contributes ~1% when isolated (mean pooling ablation: +587% RMSE).

The VM neuron is mathematically designed for angular/periodic inputs (`s = exp(κ · cos(θ + μ))`), yet it never receives angular representations of time. This is the core missed opportunity.

## Goals

1. **Teach the VM encoder about time** by feeding explicit angular temporal features as native VM inputs
2. **Multi-scale periodic decomposition** via a frequency bank with frozen κ bands
3. **Reduce LSTM dependency** through depthwise temporal convolutions between VM layers

## Non-Goals

- Replacing the LSTM entirely (incremental reduction, not elimination)
- Modifying the existing `VonMisesForecaster` or `VonMisesLinear` API beyond adding an optional `phase_shift` parameter
- Changing the data pipeline (`preprocessing.py`, `sequences.py`, `split.py`)

---

## Architecture

### 1. Angular Time Extraction

**Function:** `extract_angular_time_features(index, periods) → np.ndarray`

Converts a DatetimeIndex into raw angles in [0, 2π):

| Feature | Formula | Period |
|---------|---------|--------|
| θ_hour | 2π · hour / 24 | Daily (24h) |
| θ_dow | 2π · (dow·24 + hour) / 168 | Weekly (168h) |
| θ_season | 2π · day_of_year / 365.25 | Annual (8766h) |

**Key design choice:** Raw angles, not sin/cos pairs. The VM neuron learns its own decomposition via `κ · cos(θ + μ)`, discovering the optimal phase shift μ. Pre-decomposing into sin/cos would bypass the neuron's native capability.

**Output:** `(T, n_periods)` array. Concatenated with sensor features as additional columns before `create_sequences()`.

**Normalization bypass:** Angular time features must NOT be z-score normalized — z-scoring maps [0, 2π) to an arbitrary centered range, destroying the angular interpretation that VM neurons depend on. In the data pipeline, `extract_angular_time_features()` is called *after* `normalize()` and the resulting columns are concatenated to the already-normalized sensor DataFrame. This keeps angular features in [0, 2π) while sensor features are z-scored as usual. The pipeline order is: `chronological_split() → normalize(sensor_cols) → extract_angular_time_features() → concat → create_sequences()`.

**Configurable periods:** Default `[24, 168, 8766]` for hourly data. Arbitrary periods supported for other resolutions.

**Off-by-one note:** `day_of_year` starts at 1 in Python, so θ_season ≈ 0.017 rad on Jan 1st, not exactly 0. This is harmless — the VM neuron learns μ to compensate — but implementers should not assume 0-anchoring at year boundaries.

### 2. Frequency Bank VM Layer

**Class:** `FrequencyBankVonMises(nn.Module)`

A VM layer where neurons are organized into frequency bands with **frozen κ** and **learnable μ**.

**Constructor:** `FrequencyBankVonMises(in_features, out_features, freq_bands)` where `in_features = len(time_periods)` — each input column is one raw angular time feature from Section 1.

**Default bands (hourly data):**

| Band | κ (frozen) | Window | Responds to |
|------|-----------|--------|-------------|
| Sharp | 8.0 | ~±0.35 rad | Hourly/sub-daily |
| Medium | 2.0 | ~±1.0 rad | Daily-scale |
| Broad | 0.5 | ~±2.5 rad | Weekly/seasonal |

**Neuron allocation:** Given `out_features` and N bands, neurons are split equally (remainder distributed to first bands). Each neuron in a band shares its frozen κ but learns its own μ.

**Why freeze κ:** Without freezing, gradient descent collapses all neurons toward the dominant timescale's optimal κ, losing the multi-scale property. Frozen κ forces the network to use all timescales. μ learns freely within each band.

**Frozen κ implementation:** Store κ values as a `register_buffer` (not `nn.Parameter`). This ensures:
- κ is excluded from `model.parameters()` and optimizer param groups (no wasted memory for moment buffers)
- κ moves to GPU correctly with `.to(device)` (unlike a plain tensor attribute)
- κ appears in `state_dict()` for checkpoint saving/loading
- `FrequencyBankVonMises` internally constructs a `(out_features,)` buffer where each entry is set to its band's κ value. The `forward()` expands this to `(out_features, in_features)` via broadcasting (all input dimensions share the same κ for a given neuron, matching the band-level semantics).

**Interpretability:** After training, inspect each band:
- "Sharp band neurons peaked at hours 8, 12, 18 → learned occupancy times"
- "Broad band neurons peaked at day 2, 5 → learned weekday/weekend"

### 3. Time-Conditioned Phase Fusion

**Mechanism:** The frequency bank output modulates the sensor VM encoder's phase shifts.

```
Standard VM:        z_j = κⱼᵢ · cos(θ_sensor_i + μⱼᵢ)
Time-conditioned:   z_j = κⱼᵢ · cos(θ_sensor_i + μⱼᵢ + Δμⱼ(t))
```

Where `Δμ(t) = Linear(freq_bank_output)`, shape `(B, T, out_features, 1)`.

**Phase shift shape and broadcasting:** The phase shift is a *per-neuron* correction (not per neuron-input pair). The projection `Linear(H_time → out_features)` produces `(B, T, out_features)`, which is then unsqueezed to `(B, T, out_features, 1)`. This broadcasts against `self.mu` of shape `(out_features, in_features)` — the same Δμ shifts all input dimensions of a given neuron equally. This is the natural interpretation: "shift neuron j's preferred direction by Δμⱼ at time t."

**Periodicity note:** `cos(θ + μ + Δμ)` is 2π-periodic in θ when θ is an angle. For the *time* features entering through the frequency bank, this guarantee holds fully. For z-scored sensor features, the inputs are not raw angles, so 2π-periodicity in the sensor dimension is not meaningful — the VM layer acts as a learned nonlinearity on those dimensions. The phase fusion still works correctly: Δμ shifts the sensor neuron's preferred response point on the z-scored scale, which is the intended behavior. Multiplicative scaling of κ would also preserve periodicity; phase addition is chosen because it is the most interpretable (a time-dependent shift of the neuron's "preferred value").

**Concrete meaning:** A sensor neuron that peaks at CO2 ≈ 600ppm can learn "peak at 600ppm in the morning, shift to 580ppm at night" — the time pathway slides its preferred direction.

**Implementation:** `VonMisesLinear.forward()` gains an optional `phase_shift` parameter:

```python
def forward(self, x: Tensor, phase_shift: Tensor = None) -> Tensor:
    kappa = F.softplus(self.log_kappa)  # (out, in)
    # phase_shift: (..., out, 1) — per-neuron, broadcasts across in_features
    effective_mu = self.mu if phase_shift is None else self.mu + phase_shift
    cos_term = torch.cos(x.unsqueeze(-2) + effective_mu)  # (..., out, in)
    z = (kappa * cos_term).sum(dim=-1)  # (..., out)
    ...
```

Backward-compatible: existing callers pass no `phase_shift` and get identical behavior. The `log_output` and `bias` logic that follows `z` is unchanged.

### 4. Temporal Convolution Bridges

**Purpose:** Local temporal mixing between VM layers to reduce LSTM dependency.

```
Per VM layer:
  VonMisesLinear(phase_shift=Δμᵢ)
  → LayerNorm
  → DepthwiseConv1d(kernel=3, groups=H)
  → Residual connection (skip around VM+Conv)
  → Dropout
```

**Why depthwise conv:**
- Each channel mixes with itself across 3 adjacent timesteps (no cross-channel mixing — that's the VM layer's job)
- Parameter cost: ~3H per conv layer (tiny)
- Stacking N layers gives receptive field of 2N+1 timesteps

**First-layer residual:** When `sensor_size ≠ hidden_size`, the residual skip has a shape mismatch. The first VM block uses a `nn.Linear(sensor_size, hidden_size)` projection on the skip path (same pattern as `DeepVMPoolForecaster` in `vm_variants.py`). Subsequent layers have `hidden_size → hidden_size` and use identity skip.

**Ablation flag:** `use_temporal_conv: bool = True`. When False, no conv bridges and no residual connections — encoder matches Section 3 exactly.

**Goal:** Shift the temporal modeling balance from LSTM=99% toward LSTM≈70%, VM+conv≈30%.

### 5. Complete Model: TemporalVonMisesForecaster

```
Input: (B, T, F_sensor + F_time)    [time features precomputed]
  ↓ split on sensor_size boundary
Sensor features (B, T, F_sensor)     Angular time (B, T, F_time)
  ↓                                    ↓
  ↓                                  FrequencyBankVonMises
  ↓                                    ↓ (B, T, H_time)
  ↓                                  Linear → Δμ₁ (B, T, H, 1)
  ↓                                    ↓
VonMisesLinear(sensor, phase_shift=Δμ₁)    [layer 1: sensor_size → hidden_size]
  → LayerNorm → Conv1d → Residual → Dropout
  ↓
VonMisesLinear(hidden, phase_shift=Δμ₂)    [layer 2+: hidden_size → hidden_size]
  → LayerNorm → Conv1d → Residual → Dropout
  ↓ [repeat × num_vm_layers total, each with own Δμ projection]
  ↓ (B, T, H)

Each Δμ projection: Linear(H_time → hidden_size) → unsqueeze(-1) → (B, T, hidden_size, 1)
The projection input is always the same freq_bank_output (B, T, H_time).
The projection output matches out_features=hidden_size for all layers (layer 1 projects
sensor_size→hidden_size, subsequent layers project hidden_size→hidden_size, but
the Δμ projection always targets out_features=hidden_size).
LSTM backbone (1 layer)
  ↓ (B, H)
Linear output head → (B, output_size)
```

**Constructor:**

```python
TemporalVonMisesForecaster(
    sensor_size: int,
    time_periods: List[int],          # e.g. [24, 168, 8766]
    hidden_size: int = 64,
    num_vm_layers: int = 2,
    freq_bands: List[float] = [0.5, 2.0, 8.0],
    dropout: float = 0.0,
    output_size: int = 1,
    use_temporal_conv: bool = True,
    conv_kernel_size: int = 3,
    use_lstm: bool = True,
)
```

**Timestamp handling:** Option (a) — precompute angular features in the data pipeline via `extract_angular_time_features()`, concatenate to sensor tensor before `create_sequences()`. The model's `forward()` stays tensor-in/tensor-out. `sensor_size` tells the model where the split is. Feature ordering convention: sensor columns first (indices `[0, sensor_size)`), angular time columns last (indices `[sensor_size, sensor_size + len(time_periods))`).

**`use_lstm=False` behavior:** When `use_lstm=False`, the LSTM backbone is replaced by mean pooling over the time dimension: `h_final = h.mean(dim=1)`, producing `(B, H)`. This matches the existing `VonMisesForecaster` ablation behavior and isolates the VM encoder + conv bridge contribution.

**Device handling:** All new modules follow the standard `.to(device)` pattern. The `FrequencyBankVonMises` frozen κ is a `register_buffer`, which moves to GPU correctly. No special device logic needed.

---

## File Impact

### New file
- **`src/temporal.py`** — All new components:
  - `extract_angular_time_features()`
  - `FrequencyBankVonMises`
  - `TemporalVonMisesEncoder`
  - `TemporalVonMisesForecaster`

### Modified files
- **`src/von_mises_neuron.py`** — Add optional `phase_shift` parameter to `VonMisesLinear.forward()`. Backward-compatible.
- **`scripts/benchmark_standard_dataset.py`** — Add `benchmark_temporal_von_mises()`, include in comparison table. The benchmark pipeline for this model: generate synthetic data → `chronological_split()` → `normalize()` (sensor columns only) → `extract_angular_time_features()` on each split's DatetimeIndex → concatenate angular columns to normalized sensor DataFrame → `create_sequences()`. Pass `sensor_size=4` (Temperature, Humidity, CO2, Noise) to the model constructor.
- **`tests/`** — New `tests/test_temporal.py` for all new components.

### Untouched files
- `src/preprocessing.py`, `src/sequences.py`, `src/split.py`
- `src/baselines.py`, `src/vm_variants.py`, `src/circular.py`
- `src/angular_data.py`, `src/data_loading.py`

---

## Test Plan

### Unit tests (`tests/test_temporal.py`)
1. `extract_angular_time_features()` — correct angles for known timestamps, all values in [0, 2π), correct shape
2. `FrequencyBankVonMises` — κ values frozen after init, μ values learnable, output shape, neuron allocation across bands
3. Phase fusion — `VonMisesLinear` with `phase_shift` produces different output than without, periodicity still holds with phase_shift applied
4. `TemporalVonMisesEncoder` — forward shape, gradient flow to all parameters (μ, κ where learnable, Δμ projection, conv weights)
5. `TemporalVonMisesForecaster` — full forward/backward, sensor/time split correctness

### Ablation tests
6. `use_temporal_conv=True` vs `False` — both produce valid output, different parameter counts
7. `use_lstm=True` vs `False` — mean pool fallback works
8. Different `freq_bands` configurations — single band, many bands

### Pipeline tests
9. Angular features survive pipeline — verify angular columns are in [0, 2π) after full pipeline (split → normalize sensor cols → extract angles → concat → sequence)
10. Sensor/time split correctness — model receives correct columns in each pathway

### Integration / benchmark
11. `benchmark_temporal_von_mises()` — must pass mandatory thresholds (RMSE<35, MAE<25, R²>0.20, MAPE<5%)
12. Comparison against existing `VonMisesForecaster` on same synthetic data

---

## Success Criteria

1. All unit and ablation tests pass
2. Mandatory benchmark thresholds met
3. `TemporalVonMisesForecaster` matches or outperforms `VonMisesForecaster` on synthetic data — specifically, must not regress more than 10% on any metric vs. the original model's benchmark scores (not merely the minimum thresholds)
4. Ablation: `use_lstm=False` performs better than the original `VonMisesForecaster` with `use_lstm=False` (demonstrating the temporal conv bridges help)
5. Learned frequency bank phases are interpretable (sharp band aligns with known daily peaks)
