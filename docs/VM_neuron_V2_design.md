  #Proposed Approaches
  1. A: Teach VM about time directly (explicit temporal inputs)
  2. C: Multi-scale periodic decomposition (hierarchy of timescales)
  3. B: Reduce LSTM dependency (VM handles temporal dynamics itself)

  These build on each other nicely — A is foundational, C extends it, and B is the ambitious capstone. Let me propose        
  approaches for each layer.

  ---
  Approach 1: "Angular Time Injection" (Conservative)

  Feed explicit angular time features as VM inputs rather than as hand-crafted sin/cos columns.

  How it works:
  - Convert timestamps to angles: θ_hour = 2π · hour/24, θ_dow = 2π · dow/7, θ_month = 2π · month/12
  - Concatenate these raw angles with sensor features as VM layer input
  - The VM neuron naturally handles them: κ · cos(θ_hour + μ) learns phase-locked relationships

  For multi-scale (C): Add multiple periods as separate angle inputs (θ₂₄, θ₁₆₈, θ₈₇₆₆).

  For LSTM reduction (B): No direct contribution — LSTM still needed for sequence dynamics.

  Trade-offs:
  - (+) Dead simple, no architecture changes needed
  - (+) VM neurons are designed for angular inputs — this plays to their strength
  - (+) Interpretable: inspect μ to see "this neuron peaks at 3pm on Tuesdays"
  - (-) Doesn't address temporal ordering (VM still processes timesteps independently)
  - (-) Doesn't reduce LSTM dependency at all

  ---
  Approach 2: "Temporal VM Encoder" (Moderate)

  A new TemporalVonMisesEncoder that fuses angular time embeddings with sensor data through dedicated VM pathways, plus a    
  frequency-bank mechanism for multi-scale decomposition.

  How it works:
  - Time pathway: Raw angles (θ_hour, θ_dow, θ_season) processed through their own VM sub-layer
  - Sensor pathway: Raw sensor features through existing VM sub-layer
  - Fusion: Cross-product interaction — each sensor neuron's phase μ is conditioned on the time pathway output: μ_effective =   μ_base + f(time_embedding)
  - Frequency bank (C): A set of VM neurons with fixed, frozen κ values spanning octaves (κ=0.5 for broad/seasonal, κ=8 for  
  sharp/hourly). Only μ and fusion weights are learned.
  - For LSTM reduction (B): Add lightweight temporal convolutions (depthwise Conv1d) between VM layers — similar to what     
  DeepVMPoolForecaster already does, but with the time-conditioned phases.

  Trade-offs:
  - (+) Time-conditioned phases are novel: "this neuron peaks at different angles depending on time-of-day"
  - (+) Frequency bank gives explicit multi-scale decomposition with interpretable bands
  - (+) Conv1d bridges provide some LSTM-free temporal modeling
  - (-) More parameters and complexity
  - (-) Fusion mechanism needs careful design to avoid breaking periodicity guarantees

  ---
  Approach 3: "Phase-Time Attention" (Ambitious)

  Extend VonMisesAttentionForecaster with temporal phase encoding, making the attention mechanism itself
  time-and-period-aware.

  How it works:
  - Temporal phase encoding: Instead of standard sinusoidal PE, inject angular time features into the VM attention scores:   
  score_ij = Σ κ_d · cos(Q_id + μ_d - K_jd + φ_time(i,j)) where φ_time encodes the temporal distance between positions i and 
  j as an angle
  - Multi-scale (C): Multiple attention heads specialize at different periods — head 1 attends at daily scale, head 2 at     
  weekly, etc. Achieved by initializing each head's κ/μ at different frequency bands
  - LSTM replacement (B): The attention mechanism is the temporal model — no LSTM needed

  Trade-offs:
  - (+) Fully periodic temporal model — VM all the way down
  - (+) Attention provides interpretable "which timesteps matter" maps
  - (+) Natural multi-scale via multi-head specialization
  - (-) Most complex to implement and debug
  - (-) VonMisesAttentionForecaster already exists but hasn't been benchmarked — unknown baseline
  - (-) Attention is O(T²) vs LSTM's O(T)

  ---
 #Retained Approach

  Approach 2 (Temporal VM Encoder), with elements of Approach 1 as the starting layer.

  Reasoning:
  - Approach 1 alone is too conservative — it doesn't address your B or C priorities at all
  - Approach 3 is exciting but risky — you'd be debugging periodic attention and temporal features simultaneously
  - Approach 2 gives you a clean progression:
    a. Start with angular time injection (A — immediate win, testable)
    b. Add frequency bank for multi-scale (C — modular addition)
    c. Add time-conditioned phases + Conv1d bridges (B — reduces LSTM dependency incrementally)
  - Each step is independently testable against the mandatory benchmark
  - The frequency bank idea is particularly compelling because it makes the multi-scale decomposition inspectable — you can  
  literally read off "this neuron responds to the daily cycle, that one to the weekly"

---
● Section 1: Angular Time Extraction

  Purpose: Convert timestamps into raw angles that VM neurons can natively consume.

  New module: src/temporal.py

  Core function: extract_angular_time_features()

  Takes a DatetimeIndex and produces a tensor of angular features:

  ┌─────────┬─────────────────────────────┬────────┐
  │ Feature │           Formula           │ Period │
  ├─────────┼─────────────────────────────┼────────┤
  │ θ_hour  │ 2π · hour / 24              │ Daily  │
  ├─────────┼─────────────────────────────┼────────┤
  │ θ_dow   │ 2π · dayofweek / 7          │ Weekly │
  ├─────────┼─────────────────────────────┼────────┤
  │ θ_month │ 2π · (day_of_year / 365.25) │ Annual │
  └─────────┴─────────────────────────────┴────────┘

  These are raw angles in [0, 2π) — not sin/cos pairs. This is the key difference from augment_with_time_features() in the   
  benchmark script, which pre-decomposes into sin/cos. By feeding raw angles, we let the VM neuron learn its own
  decomposition: κ · cos(θ_hour + μ) discovers the optimal phase shift μ, which is exactly what the neuron is built to do.   

  Output shape: (T, n_periods) — one column per period. These get concatenated with sensor features before entering the VM   
  encoder.

  Configurable periods: Default [24, 168, 8766] (hourly data: daily, weekly, yearly), but user can pass arbitrary period list   for different resolutions.
● Section 2: Frequency Bank VM Layer

  Purpose: Explicit multi-scale periodic decomposition. Different neurons specialize at different timescales by design, not  
  by accident.

  New class: FrequencyBankVonMises(nn.Module) in src/temporal.py

  Core idea: A VM layer where neurons are organized into frequency bands. Each band has a fixed, frozen κ that controls its  
  temporal resolution, while μ (phase) remains learnable.

  Band structure (default for hourly data):

  ┌────────┬────────────┬──────────────────────────────────┬───────────────────────────┐
  │  Band  │ κ (frozen) │          Interpretation          │        Responds to        │
  ├────────┼────────────┼──────────────────────────────────┼───────────────────────────┤
  │ Sharp  │ 8.0        │ Narrow peaks, ~±0.35 rad window  │ Hourly/sub-daily patterns │
  ├────────┼────────────┼──────────────────────────────────┼───────────────────────────┤
  │ Medium │ 2.0        │ Moderate peaks, ~±1.0 rad window │ Daily-scale patterns      │
  ├────────┼────────────┼──────────────────────────────────┼───────────────────────────┤
  │ Broad  │ 0.5        │ Wide peaks, ~±2.5 rad window     │ Weekly/seasonal trends    │
  └────────┴────────────┴──────────────────────────────────┴───────────────────────────┘

  How neurons are allocated: Given out_features=64 and 3 bands, allocate neurons proportionally (configurable, default equal:   22/21/21). Each neuron in a band shares the same frozen κ but learns its own μ.

  Why freeze κ?
  - Without freezing, gradient descent collapses all neurons toward the same κ that minimizes training loss — you lose the   
  multi-scale property
  - Frozen κ forces the network to use all timescales, even if one dominates the training signal
  - μ still learns freely, so each neuron within a band picks its preferred phase angle

  Mathematically:
  z_j = κ_band(j) · cos(θᵢ + μⱼᵢ)     [κ frozen per band, μ learned]

  Interpretability bonus: After training, you can inspect each band independently:
  - "Sharp band neurons peaked at hours 8, 12, 18 → learned meal/occupancy times"
  - "Broad band neurons peaked at day 2, 5 → learned weekday/weekend distinction"

  Integration with the rest of the model: FrequencyBank processes the angular time features from Section 1. Its output gets  
  fused with sensor features (fusion is Section 3). It does NOT replace the existing VM encoder — it's a parallel pathway for   temporal structure.

● Section 3: Time-Conditioned Fusion

  Purpose: Connect the temporal pathway (FrequencyBank output) to the sensor pathway (existing VM encoder) so that the VM    
  encoder's behavior adapts based on what time it is.

  Mechanism: Learned phase modulation

  The key insight: instead of just concatenating time and sensor embeddings, we let the time pathway shift the phase of the  
  sensor VM neurons. This is mathematically natural — phase shift is exactly what μ controls.

  How it works:

  Standard VM:        z_j = κⱼᵢ · cos(θ_sensor_i + μⱼᵢ)
  Time-conditioned:   z_j = κⱼᵢ · cos(θ_sensor_i + μⱼᵢ + Δμⱼ(t))

  Where Δμⱼ(t) is a small phase correction produced by a linear projection from the frequency bank output:

  Δμ = Linear(freq_bank_output)    shape: (B, T, out_features)

  Why this preserves periodicity: Adding Δμ inside the cosine doesn't break 2π-periodicity — cos(θ + μ + Δμ) is still        
  periodic in θ. This is the only fusion strategy that maintains the VM guarantee. Multiplying or gating would break it.     

  New class: TemporalVonMisesEncoder(nn.Module) in src/temporal.py

  This wraps the full pipeline:

  Input: (B, T, F_sensor + F_time)
    ↓ split
  Sensor features (B, T, F_sensor)     Angular time features (B, T, F_time)
    ↓                                    ↓
    ↓                                  FrequencyBankVonMises
    ↓                                    ↓ (B, T, H_time)
    ↓                                  Linear → Δμ (B, T, H_sensor)
    ↓                                    ↓
  VonMisesLinear(sensor, phase_shift=Δμ) ← fused here
    ↓ LayerNorm → Dropout
    ↓ [repeat N times]
    ↓
  Output: (B, T, H_sensor)

  Key design decisions:
  - Δμ is not clamped — the cosine wraps naturally, so any magnitude is fine
  - The linear projection from H_time → H_sensor is the only new learned parameter in the fusion (lightweight)
  - Each stacked VM layer gets its own Δμ projection (time conditioning applies at every depth)

  What this means concretely: A sensor VM neuron that normally peaks when CO2 ≈ 600ppm can now learn "peak at 600ppm during  
  morning hours, but shift to 580ppm at night" — the time pathway slides its preferred direction.

● Section 4: Reducing LSTM Dependency

  Purpose: Give the TemporalVonMisesEncoder some ability to model temporal dynamics across timesteps, so the LSTM isn't      
  shouldering 99% of the temporal work.

  Mechanism: Depthwise temporal convolutions between VM layers

  This borrows from DeepVMPoolForecaster in vm_variants.py, which already uses Conv1d(groups=d_model) for local temporal     
  mixing. The difference: here it's combined with time-conditioned phases, so the conv bridges operate on richer per-timestep   representations.

  Where it fits in the encoder stack:

  For each VM layer i:
    VonMisesLinear(phase_shift=Δμᵢ)
    → LayerNorm
    → DepthwiseConv1d(kernel=3, groups=H)   ← local temporal mixing
    → Residual connection                     ← skip around VM+Conv block
    → Dropout

  Why depthwise conv, not full conv or attention:
  - Depthwise (groups=H): Each channel mixes with itself across 3 adjacent timesteps. No cross-channel mixing — that's the VM   layer's job. This keeps the parameter count tiny (~3H params per conv layer).
  - Kernel=3: Captures immediate neighbors (t-1, t, t+1). Stacking N VM layers gives a receptive field of 2N+1 timesteps —   
  with N=3, that's 7 hours of local context without LSTM.
  - Residual: Ensures the conv bridge is additive. If it's not useful, gradients push its weights toward zero and the model  
  degrades gracefully to per-timestep processing.

  What this does NOT do: Replace the LSTM entirely. The conv bridges capture local temporal structure (adjacent hours), while   the LSTM still handles global dependencies (e.g., "yesterday's pattern predicts today"). The goal is to shift the balance 
  — instead of LSTM=99%, aim for something like LSTM=70%, VM+conv=30%.

  Ablation control: use_temporal_conv: bool = True flag. When False, the encoder matches Section 3 exactly (no conv bridges).   This lets you measure the conv contribution independently.

  ---
  Section 5: The Complete Model — TemporalVonMisesForecaster

  Purpose: Drop-in replacement for VonMisesForecaster that uses all of the above.

  New class: TemporalVonMisesForecaster(nn.Module) in src/temporal.py

  Input: (B, T, F_sensor + F_time)
    ↓
  TemporalVonMisesEncoder (Sections 1-4)
    [FrequencyBank + Phase Fusion + Conv Bridges] × N layers
    ↓ (B, T, H)
  LSTM backbone (1 layer, same as current)
    ↓ (B, H)
  Linear output head
    ↓ (B, output_size)

  Constructor signature:

  TemporalVonMisesForecaster(
      sensor_size: int,           # number of sensor features (no time cols)
      time_periods: List[int],    # e.g. [24, 168, 8766]
      hidden_size: int = 64,
      num_vm_layers: int = 2,
      freq_bands: List[float] = [0.5, 2.0, 8.0],  # κ per band
      dropout: float = 0.0,
      output_size: int = 1,
      use_temporal_conv: bool = True,
      conv_kernel_size: int = 3,
      use_lstm: bool = True,      # ablation: mean pool if False
  )

  Key difference from VonMisesForecaster: The model takes sensor_size (raw features only) and time_periods separately. It    
  internally calls extract_angular_time_features() — the caller does NOT need to pre-compute time columns. This means the    
  data pipeline stays unchanged except for passing the DatetimeIndex alongside the sensor tensor.

  Wait — how does it get timestamps at inference? Two options:

  - (a) Precompute angular features in the data pipeline and concatenate to the input tensor (simpler, matches current       
  create_sequences() flow)
  - (b) Pass timestamps as a separate tensor alongside X

  I'd recommend (a) — precompute in the pipeline. It keeps forward() a simple tensor-in/tensor-out, and the angular features 
  are deterministic from the timestamps anyway. The extract_angular_time_features() function becomes a preprocessing step,   
  called once per split.

  Benchmark integration: The existing benchmark_standard_dataset.py gets a new function benchmark_temporal_von_mises() that: 
  1. Calls extract_angular_time_features() on the synthetic data
  2. Concatenates angular columns to sensor features
  3. Passes sensor_size=4 (original features) and the augmented tensor to TemporalVonMisesForecaster
  4. Must pass the same acceptance thresholds (RMSE<35, MAE<25, R²>0.20, MAPE<5%)
