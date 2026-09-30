1. A: Teach VM about time directly (explicit temporal inputs)
  2. C: Multi-scale periodic decomposition (hierarchy of timescales)
  3. B: Reduce LSTM dependency (VM handles temporal dynamics itself)

  These build on each other nicely — A is foundational, C extends it, and B is the ambitious capstone. Let me propose approaches for each layer.

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

  A new TemporalVonMisesEncoder that fuses angular time embeddings with sensor data through dedicated VM pathways, plus a frequency-bank mechanism for multi-scale decomposition.

  How it works:
  - Time pathway: Raw angles (θ_hour, θ_dow, θ_season) processed through their own VM sub-layer
  - Sensor pathway: Raw sensor features through existing VM sub-layer
  - Fusion: Cross-product interaction — each sensor neuron's phase μ is conditioned on the time pathway output: μ_effective = μ_base + f(time_embedding)
  - Frequency bank (C): A set of VM neurons with fixed, frozen κ values spanning octaves (κ=0.5 for broad/seasonal, κ=8 for sharp/hourly). Only μ and fusion weights are learned.
  - For LSTM reduction (B): Add lightweight temporal convolutions (depthwise Conv1d) between VM layers — similar to what DeepVMPoolForecaster already does, but with the time-conditioned phases. 

  Trade-offs:
  - (+) Time-conditioned phases are novel: "this neuron peaks at different angles depending on time-of-day"
  - (+) Frequency bank gives explicit multi-scale decomposition with interpretable bands
  - (+) Conv1d bridges provide some LSTM-free temporal modeling
  - (-) More parameters and complexity
  - (-) Fusion mechanism needs careful design to avoid breaking periodicity guarantees

  ---
  Approach 3: "Phase-Time Attention" (Ambitious)

  Extend VonMisesAttentionForecaster with temporal phase encoding, making the attention mechanism itself time-and-period-aware.

  How it works:
  - Temporal phase encoding: Instead of standard sinusoidal PE, inject angular time features into the VM attention scores: score_ij = Σ κ_d · cos(Q_id + μ_d - K_jd + φ_time(i,j)) where φ_time   
  encodes the temporal distance between positions i and j as an angle
  - Multi-scale (C): Multiple attention heads specialize at different periods — head 1 attends at daily scale, head 2 at weekly, etc. Achieved by initializing each head's κ/μ at different       
  frequency bands
  - LSTM replacement (B): The attention mechanism is the temporal model — no LSTM needed