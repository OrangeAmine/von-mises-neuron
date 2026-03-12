# Von Mises Neuron — Project Report

## 1. The Problem: Standard Neurons Break Periodicity

A standard neuron computes `s = f(W·θ + b)`. When the input `θ` is an angle
(radians), multiplying by `W` inside the nonlinearity changes the period. A
signal that was 2π-periodic becomes `2π/W`-periodic — the neuron has
destroyed the structure it was supposed to learn from.

This matters for any domain with circular or periodic inputs: wind direction,
joint angles in robotics, time-of-day features, protein dihedral angles,
phase in signal processing.

The Von Mises neuron addresses this by placing the learnable weight
**outside** the trigonometric function, preserving the input's natural
periodicity by construction.


## 2. How It Works

### 2.1 Single Neuron

The Von Mises neuron computes:

```
s = exp(κ · cos(θ + μ))
```

where:
- **θ** is the input (an angle in radians)
- **μ ∈ ℝ** is a learnable phase shift (the neuron's "preferred direction")
- **κ > 0** is a learnable concentration (controls peak sharpness, analogous
  to 1/σ² in a Gaussian)

This is exactly the von Mises probability density function (the "circular
Gaussian") up to a normalizing constant. The output is:
- Strictly **2π-periodic** in θ — `f(θ) = f(θ + 2π)` for all θ
- Always **positive** (exp of a bounded quantity)
- **Peaked** at θ = −μ (where cos(θ + μ) = 1)
- Controlled by κ: high κ → sharp peak, low κ → broad response

### 2.2 Multi-Dimensional Extension (VonMisesLinear)

For D input dimensions and J output neurons:

```
z_j = Σᵢ κⱼᵢ · cos(θᵢ + μⱼᵢ)     (log-space)
s_j = exp(z_j) + b_j               (exp-space)
```

Each output neuron j has its own (κⱼᵢ, μⱼᵢ) pair per input dimension i.
The log-space form `z_j` is a sum of cosines — a product of independent
von Mises distributions in log-space. Each dimension independently preserves
2π-periodicity.

**Parameter count:** A VonMisesLinear(in, out) has `2 × in × out` parameters
for (κ, μ) plus an optional `out`-dimensional bias — comparable to a standard
`nn.Linear(in, out)` which has `in × out + out`.

### 2.3 Gradient Properties

For z_j = Σ κ·cos(θ + μ) and s_j = exp(z_j):

| Parameter | Gradient | Behavior |
|-----------|----------|----------|
| ∂s/∂μⱼᵢ  | −s · κⱼᵢ · sin(θᵢ + μⱼᵢ) | Vanishes when θ aligns with −μ (at the peak) |
| ∂s/∂κⱼᵢ  | s · cos(θᵢ + μⱼᵢ) | Proportional to cos — always bounded |
| ∂s/∂θᵢ   | −s · κⱼᵢ · sin(θᵢ + μⱼᵢ) | Same form as ∂s/∂μ |

The vanishing μ-gradient at cos peaks is **desirable**: once a neuron's
phase aligns with the input pattern, it stops drifting. Meanwhile, κ
still receives a gradient proportional to cos, so it can always sharpen
or soften the response.


## 3. Implementation

### 3.1 VonMisesLinear Layer (`src/von_mises_neuron.py:55`)

The drop-in replacement for `nn.Linear`:

- **Parameters:** `mu` (out × in) and `log_kappa` (out × in)
- **κ positivity:** Stored in unconstrained log-space; `softplus(log_kappa)`
  at forward time ensures κ > 0 without clamping or projection
- **Initialization:** μ sampled uniformly in [0, 2π) for full angular
  coverage; κ initialized near a target value (default 1.0) via the
  inverse softplus: `log(exp(κ_target) − 1)`
- **Output modes:**
  - **exp-space** (default): `exp(z) + bias` — the actual VM response
  - **log-space** (`log_output=True`): `z + bias` — suitable for stacking
    layers (avoids exp explosion, keeps values in a range where the next
    cos() uses its full curvature)
- **Interpretability accessors:** `get_concentrations()` returns the
  positive κ matrix; `get_phases()` returns μ wrapped to [0, 2π)

### 3.2 VonMisesForecaster Model (`src/von_mises_neuron.py:179`)

A complete forecasting architecture with separation of concerns:

```
Input (B, T, F)
  → [VonMisesLinear(log) → LayerNorm → Dropout] × N    (VM encoder)
  → LSTM(hidden_size, 1 layer)                          (temporal backbone)
  → Linear(hidden_size → horizon)                       (output head)
```

**Design rationale:**
- The VM encoder processes each timestep independently, extracting periodic
  feature representations. It answers: "what periodic patterns exist here?"
- The LSTM processes the encoded sequence temporally. It answers: "how do
  these patterns evolve over time?"
- This separation lets each component do what it's good at, rather than
  forcing the LSTM to jointly learn both periodicity and dynamics.

**Key architectural choices:**
- VM layers use **log-space** output to avoid exp explosion between layers
- **LayerNorm** after each VM layer compensates for scale inconsistencies
  (neurons with higher κ produce larger-magnitude log outputs)
- Single-layer LSTM (dropout handled by the VM encoder)
- Direct multi-step output: all horizon steps in one forward pass


## 4. Key Design Decisions

### 4.1 Softplus for κ (not exp, not clamp)

`κ = softplus(raw) = log(1 + exp(raw))`

- **Why not exp?** `exp(raw)` can overflow for large raw values, and its
  gradient `exp(raw)` grows unboundedly — training instability.
- **Why not clamp?** `clamp(raw, min=ε)` has zero gradient below ε,
  creating dead zones where κ can't recover.
- Softplus is smooth, strictly positive, and has gradients in (0, 1) —
  stable and informative everywhere.

### 4.2 Uniform μ Initialization

Phases are initialized uniformly in [0, 2π) rather than near zero. This
ensures different neurons "look at" different angular regions from the
start, preventing them from collapsing to the same phase and wasting
capacity.

### 4.3 Log-Space Stacking

Intermediate VM layers output `z = Σ κ·cos(θ + μ)` rather than `exp(z)`.
This is critical for deep stacking because:
- `cos()` accepts any real input (it wraps by 2π), so log-space outputs
  are valid inputs to the next layer
- Avoids exp-of-exp explosion that would make the network numerically
  unstable after 2+ layers

### 4.4 LayerNorm Between VM Layers

Neurons with different κ values produce outputs at different scales.
Without normalization, high-κ neurons dominate low-κ neurons in the
subsequent layer. LayerNorm equalizes their influence.

### 4.5 VM Encoder + LSTM Backbone (not pure VM)

A pure stack of VM layers can capture complex periodic feature
interactions, but it processes each timestep independently — it has no
notion of temporal order. Adding an LSTM after the VM encoder gives the
model sequential memory. This hybrid design cleanly separates periodic
feature extraction from temporal dynamics.


## 5. Benchmark Performance

### 5.1 Setup

- **Signal:** `y(t) = 600 + 50·sin(2πt/24) + 15·sin(2πt/168) + N(0,5)`
  - Daily cycle (period=24h) + weekly cycle (period=168h) + Gaussian noise
  - 4 input features: Temperature, Humidity, CO2, Noise (dB)
- **Data:** 90 days at 1-hour resolution (2,160 samples)
- **Split:** 80/10/10 chronological (1,645 / 133 / 133 sequences)
- **Model:** VonMisesForecaster(input=4, hidden=64, vm_layers=2, dropout=0.1)
- **Training:** Adam, lr=0.001, batch=64, patience=15, max 50 epochs
- **Task:** 12-hour-ahead forecast from a 72-hour lookback window
- **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU

### 5.2 Results

| Metric | Result | Threshold | Margin |
|--------|--------|-----------|--------|
| RMSE   | **5.36 ppm** | ≤ 35.0 ppm | 6.5× better |
| MAE    | **4.15 ppm** | ≤ 25.0 ppm | 6.0× better |
| R²     | **0.9790** | ≥ 0.20 | Near-perfect |
| MAPE   | **0.69%** | ≤ 5.0% | 7.2× better |

**Training time:** 88.6 seconds (50 epochs max, early stopping active)

### 5.3 Analysis

The model crushes all thresholds by large margins. This is expected for
this particular benchmark — the synthetic signal is composed of pure
sinusoids, which is exactly the kind of structure VM neurons are designed
to capture. The cos(θ + μ) activation can directly lock onto the daily
and weekly periodicities in the data.

The R² of 0.979 means the model explains 97.9% of variance in the test
set. The remaining 2.1% is mostly irreducible Gaussian noise (σ=5 ppm
on a ~150 ppm signal range).

**Important caveat:** This benchmark validates that the implementation
is correct and the model can learn periodic structure. It does *not* yet
prove the VM neuron outperforms baselines on realistic data — that
requires comparative experiments (see Section 6).


## 6. Ideas to Test

### 6.1 Comparative Baselines (Priority: High)

The most important next step — prove the VM neuron adds value over
standard approaches:

- **LSTM-only baseline:** Same architecture but replace the VM encoder
  with standard Linear + ReLU layers. Same parameter count. If the VM
  neuron's periodicity preservation matters, it should outperform this
  baseline on periodic signals and match it on non-periodic ones.
- **Transformer baseline:** Encoder-only Transformer with sinusoidal
  positional encoding. Tests whether learned attention can implicitly
  capture periodicity as well as explicit VM encoding.
- **Linear + hand-crafted features:** sin(2πt/24), cos(2πt/24), etc.
  as engineered time features fed to a linear model. This is the
  "manual engineering" approach the VM neuron should automate.

### 6.2 Real-World Periodic Datasets (Priority: High)

Test on data where periodicity is natural but noisy:

- **UCI Occupancy Detection:** Indoor CO2 with daily occupancy cycles.
  Close to the synthetic benchmark but with real noise, missing data,
  and non-stationary patterns.
- **Electricity load forecasting:** Strong daily/weekly periodicity.
  Datasets like UCI Individual Household Electric Power Consumption or
  the Electricity Transformer dataset.
- **Weather data:** Temperature and wind — clear diurnal and seasonal
  cycles with complex interactions.

### 6.3 Ablation Studies (Priority: Medium)

Isolate what matters in the architecture:

- **Number of VM layers:** 1 vs. 2 vs. 3 vs. 4. Does stacking help, or
  does one layer capture all the periodicity?
- **κ initialization:** Sweep init_kappa from 0.1 (very broad) to 10.0
  (very sharp). Does it matter, or does training recover regardless?
- **Log-space vs. exp-space stacking:** The current design uses log-space
  between layers. What happens with exp-space (expected: instability)?
- **LayerNorm removal:** Confirm that removing LayerNorm between VM
  layers degrades performance (high-κ neuron domination).
- **VM encoder only (no LSTM):** Flatten the VM encoder output and feed
  directly to the linear head. This isolates the VM contribution from
  the LSTM's temporal modeling.

### 6.4 Interpretability Analysis (Priority: Medium)

The VM neuron's parameters are directly interpretable:

- **Phase alignment:** After training on daily data, do learned μ values
  cluster near the true signal peaks? (Expected: yes, and this is
  verifiable since the synthetic signal's phase is known.)
- **Concentration evolution:** Track κ during training. Does it start
  broad and sharpen, or vice versa? Do different neurons specialize on
  different input features?
- **Phase visualization:** Plot the learned VM response curves
  `exp(κ·cos(θ + μ))` for each neuron as a function of θ. This is a
  circular "receptive field" that should reveal what angular regions
  each neuron attends to.

### 6.5 Non-Periodic Data (Priority: Medium)

Test the VM neuron's behavior on data with no periodic structure:

- **Random walk / AR(1) process:** No periodicity at all. The VM neuron
  should not *hurt* here (it can learn broad κ → flat response), but it
  shouldn't help either.
- **Trend-dominated data:** Stock prices, monotonic growth signals. Does
  the VM encoder learn to "pass through" non-periodic patterns, or does
  it distort them?

### 6.6 Pure VM Architectures (Priority: Low)

Explore removing the LSTM entirely:

- **Deep VM stack + pooling:** Multiple VM layers → mean pool → linear.
  Can the VM encoder alone capture temporal dynamics through phase
  interactions across stacked layers?
- **VM attention:** Use VM kernels as attention scores —
  `attn(q, k) = exp(κ·cos(q + μ − k))` instead of dot-product
  attention. This would make a "periodic Transformer" that natively
  handles circular relationships.

### 6.7 Other Periodic Domains (Priority: Low)

The VM neuron is domain-agnostic. Testing on non-time-series circular data:

- **Wind direction forecasting:** Input is already in radians.
- **Protein structure:** Dihedral angles (φ, ψ) in the Ramachandran plot.
- **Robotics joint control:** Motor angles with wrap-around at 2π.


## 7. Test Coverage

25 unit tests covering both `VonMisesLinear` and `VonMisesForecaster`:

| Category | Tests | What's Verified |
|----------|-------|-----------------|
| Output shapes | 5 | 2D, 3D inputs; various batch/feature/horizon sizes |
| 2π-periodicity | 3 | f(θ) == f(θ+2π) globally, per-dimension, and through full model |
| Parameter constraints | 3 | κ > 0 always, κ near init target, μ in [0, 2π) |
| Gradient flow | 3 | Gradients reach μ, log_κ, and bias |
| Output properties | 2 | Exp-space always positive; log-space == log(exp-space) |
| Peak behavior | 1 | Output peaks at θ = −μ (preferred direction) |
| Determinism | 2 | Same input → same output in eval mode |
| Dropout effect | 1 | Different outputs in train mode with dropout |
| Interpretability | 1 | get_vm_layers / get_layer_concentrations / get_layer_phases |
| Architecture variants | 2 | No-bias mode; single-layer model |
| Benchmark | 1 | End-to-end synthetic signal pass/fail (RMSE, MAE, R², MAPE) |
