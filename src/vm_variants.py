"""Experimental VM architecture variants — LSTM-free forecasters.

These models explore whether the Von Mises neuron can model temporal
dynamics *without* an LSTM backbone.  Two approaches:

1. **DeepVMPoolForecaster**: Deep residual VM stack with depthwise
   temporal convolution and mean pooling.  Tests whether depth + local
   temporal mixing can substitute for recurrence.

2. **VonMisesAttentionForecaster**: Replaces dot-product attention with
   a Von Mises kernel:

       attn(q, k) = exp(kappa * cos(q + mu - k))

   This creates a "periodic Transformer" whose attention scores are
   natively 2pi-periodic — positions with similar phase attend strongly
   regardless of angular wrapping.

Context from prior experiments:
  - Ablation study: LSTM carries ~99% of accuracy; mean pool = +587% RMSE
  - Interpretability: kappa ~ 1.0 everywhere — broad, soft extractors
  - Stress test: VM hurts on non-periodic data (+108-112% RMSE)

Goal is research exploration, not beating the LSTM.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.von_mises_neuron import VonMisesLinear
from src.baselines import SinusoidalPositionalEncoding


# ---------------------------------------------------------------------------
# Variant 1: Deep VM Stack + Pooling (no LSTM, no attention)
# ---------------------------------------------------------------------------


class DeepVMPoolForecaster(nn.Module):
    """Forecaster using a deep stack of residual VM blocks with mean pooling.

    Architecture::

        Input (B, T, F)
          -> Linear(F, d_model)                           [input projection]
          -> Learned PE: Embedding(max_len, d_model)      [temporal signal]
          -> N x DeepVMBlock:
               VonMisesLinear -> LayerNorm -> Dropout
               -> Conv1d(depthwise, K=3) -> GELU          [local temporal mix]
               -> Residual add
          -> Mean pool over T                              [aggregate time]
          -> Linear(d_model, output_size)                  [forecast head]

    The depthwise Conv1d mixes each channel with itself across adjacent
    timesteps (groups=d_model), providing a local temporal receptive field
    without cross-feature mixing (the VM layer handles that).

    Learned positional encoding (not sinusoidal) lets the model discover
    arbitrary temporal patterns without imposing periodicity assumptions.

    Args:
        input_size: Number of input features per timestep.
        d_model: Working dimension for all layers (default 64).
        num_blocks: Number of residual VM blocks (default 4).
        conv_kernel_size: Kernel size for depthwise Conv1d (default 3).
        dropout: Dropout rate (default 0.1).
        output_size: Forecast horizon (default 1).
        init_kappa: Initial concentration for VM layers (default 1.0).
        max_len: Maximum sequence length for positional encoding (default 512).
        use_residual: If True, add skip connections around each block.
        use_conv: If True, include depthwise Conv1d temporal mixing.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        num_blocks: int = 4,
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        output_size: int = 1,
        init_kappa: float = 1.0,
        max_len: int = 512,
        use_residual: bool = True,
        use_conv: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self._use_residual = use_residual
        self._use_conv = use_conv

        # --- Input projection: F -> d_model ---
        # Raw feature count (e.g. 4) is too small for residual connections.
        # Project up so skip connections and deep stacking work.
        self.input_proj = nn.Linear(input_size, d_model)

        # --- Learned positional encoding ---
        # Without temporal signal, mean pool can't distinguish timestep order.
        # Learned (not sinusoidal) avoids imposing periodicity assumptions.
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # --- Deep VM blocks ---
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.ModuleDict({
                "vm": VonMisesLinear(
                    d_model, d_model,
                    bias=True,
                    log_output=True,  # log-space for safe stacking
                    init_kappa=init_kappa,
                ),
                "ln": nn.LayerNorm(d_model),
                "dropout": nn.Dropout(dropout),
            })
            if use_conv:
                # Depthwise Conv1d: each channel mixes only with itself
                # across adjacent timesteps (groups=d_model).
                # padding=K//2 preserves sequence length.
                block["conv"] = nn.Conv1d(
                    d_model, d_model,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    groups=d_model,
                )
            self.blocks.append(block)

        # --- Output head ---
        self.output_head = nn.Linear(d_model, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        B, T, _ = x.shape

        # Project input features to working dimension
        h = self.input_proj(x)  # (B, T, d_model)

        # Add learned positional encoding
        positions = torch.arange(T, device=x.device)  # (T,)
        h = h + self.pos_embedding(positions)  # broadcast over batch

        # Deep VM blocks with optional residual + temporal conv
        for block in self.blocks:
            residual = h

            # VM periodic feature extraction (per-timestep)
            h = block["vm"](h)       # (B, T, d_model)
            h = block["ln"](h)       # normalize scale
            h = block["dropout"](h)

            if self._use_conv and "conv" in block:
                # Conv1d expects (B, C, T), our tensor is (B, T, C)
                h = h.transpose(1, 2)             # (B, d_model, T)
                h = F.gelu(block["conv"](h))      # local temporal mixing
                h = h.transpose(1, 2)             # (B, T, d_model)

            if self._use_residual:
                h = h + residual  # gradient highway

        # Mean pool over time, then project to forecast horizon
        h_final = h.mean(dim=1)  # (B, d_model)
        return self.output_head(h_final)  # (B, output_size)

    def get_vm_layers(self) -> List[VonMisesLinear]:
        """Return all VonMisesLinear layers for interpretability."""
        return [block["vm"] for block in self.blocks]

    def get_layer_concentrations(self) -> List[Tensor]:
        """Return concentration matrices for all VM layers.

        Returns:
            List of tensors, each (d_model, d_model) with kappa > 0.
        """
        return [vm.get_concentrations() for vm in self.get_vm_layers()]

    def get_layer_phases(self) -> List[Tensor]:
        """Return phase shift matrices for all VM layers.

        Returns:
            List of tensors, each (d_model, d_model) in [0, 2pi).
        """
        return [vm.get_phases() for vm in self.get_vm_layers()]


# ---------------------------------------------------------------------------
# Variant 2: Von Mises Attention (periodic Transformer)
# ---------------------------------------------------------------------------


class VonMisesMultiHeadAttention(nn.Module):
    """Multi-head attention with Von Mises kernel instead of dot product.

    Standard attention computes:
        score_ij = sum_d Q_id * K_jd / sqrt(d_k)

    VM attention replaces this with a periodic similarity:
        score_ij = sum_d kappa_d * cos(Q_id + mu_d - K_jd)

    This preserves 2pi-periodicity: shifting all Q (or K) values by 2pi
    produces identical attention weights.  Positions with similar phase
    attend strongly regardless of angular wrapping — a property that
    dot-product attention does not have.

    Each head has its own learnable kappa (concentration) and mu (phase
    bias), giving it a different "periodic receptive field" in the
    query-key space.

    Args:
        d_model: Model embedding dimension.
        nhead: Number of attention heads (must divide d_model).
        dropout: Dropout on attention weights (default 0.0).
        init_kappa: Initial concentration per head dimension (default 1.0).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        init_kappa: float = 1.0,
    ) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model={d_model} must be divisible by nhead={nhead}"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead

        # Standard linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)

        # Per-head VM parameters: kappa (concentration) and mu (phase bias)
        # kappa stored in log-space; softplus ensures kappa > 0
        self.log_kappa = nn.Parameter(torch.empty(nhead, self.d_head))
        self.mu = nn.Parameter(torch.empty(nhead, self.d_head))

        self.attn_dropout = nn.Dropout(dropout)

        self._init_kappa_target = init_kappa
        self._reset_vm_parameters()

    def _reset_vm_parameters(self) -> None:
        """Initialize VM-specific parameters.

        mu: uniform in [0, 2pi) — each head starts with different phase bias.
        kappa: near init_kappa via softplus inverse.
        """
        nn.init.uniform_(self.mu, 0, 2 * math.pi)
        if self._init_kappa_target > 0.05:
            inv = math.log(math.exp(self._init_kappa_target) - 1)
        else:
            inv = -3.0
        nn.init.constant_(self.log_kappa, inv)

    def forward(self, x: Tensor) -> Tensor:
        """Compute VM multi-head attention.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model).

        Returns:
            Output tensor, shape (batch, seq_len, d_model).
        """
        B, T, _ = x.shape

        # Linear projections, then reshape to (B, nhead, T, d_head)
        Q = self.W_q(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.nhead, self.d_head).transpose(1, 2)
        # Q, K, V: (B, nhead, T, d_head)

        # VM attention score: sum_d kappa_d * cos(Q_id + mu_d - K_jd)
        # Broadcast: Q[:,:,:,None,:] is (B, nhead, T, 1, d_head)
        #            K[:,:,None,:,:] is (B, nhead, 1, T, d_head)
        #            Difference:        (B, nhead, T, T, d_head)
        kappa = F.softplus(self.log_kappa)  # (nhead, d_head), all > 0

        # Reshape kappa and mu for 5D broadcasting: (1, nhead, 1, 1, d_head)
        kappa_5d = kappa.view(1, self.nhead, 1, 1, self.d_head)
        mu_5d = self.mu.view(1, self.nhead, 1, 1, self.d_head)

        cos_term = torch.cos(
            Q[:, :, :, None, :] + mu_5d - K[:, :, None, :, :]
        )  # (B, nhead, T, T, d_head)

        # Weighted sum over d_head dimension -> scalar attention score
        scores = (kappa_5d * cos_term).sum(dim=-1)  # (B, nhead, T, T)

        # Softmax over key positions (last dim) -> attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, nhead, T, T)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, V)  # (B, nhead, T, d_head)

        # Concat heads and project: (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_out(out)

    def get_concentrations(self) -> Tensor:
        """Return kappa matrix (nhead, d_head) with kappa > 0."""
        return F.softplus(self.log_kappa).detach()

    def get_phases(self) -> Tensor:
        """Return mu matrix (nhead, d_head) in [0, 2pi)."""
        return (self.mu.detach() % (2 * math.pi))


class VonMisesAttentionForecaster(nn.Module):
    """Forecaster using Von Mises attention (periodic Transformer).

    Architecture::

        Input (B, T, F)
          -> Linear(F, d_model)                              [input proj]
          -> SinusoidalPositionalEncoding                    [position info]
          -> N x VonMisesAttentionBlock:
               -> LayerNorm -> VM-MHA -> Dropout -> Residual [self-attention]
               -> LayerNorm -> FFN -> Dropout -> Residual    [feedforward]
          -> Mean pool over T                                [aggregate time]
          -> Linear(d_model, output_size)                    [forecast head]

    Uses pre-norm (LayerNorm before attention/FFN) for training stability,
    following the convention from GPT-2 and subsequent transformer research.

    The VM attention mechanism replaces the dot-product Q*K^T with a
    cosine kernel kappa * cos(Q + mu - K), making attention scores
    natively 2pi-periodic.

    Args:
        input_size: Number of input features per timestep.
        d_model: Model embedding dimension (default 64).
        nhead: Number of attention heads (default 4).
        num_layers: Number of attention blocks (default 2).
        dim_feedforward: FFN intermediate dimension (default 128).
        dropout: Dropout rate (default 0.1).
        output_size: Forecast horizon (default 1).
        init_kappa: Initial concentration for VM attention (default 1.0).
        max_len: Maximum sequence length (default 512).
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_size: int = 1,
        init_kappa: float = 1.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # --- Input projection ---
        self.input_proj = nn.Linear(input_size, d_model)

        # --- Positional encoding (sinusoidal, reuse from baselines) ---
        # Sinusoidal PE is parameter-free; keeps the experiment focused
        # on the attention mechanism itself.
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_len, dropout=dropout
        )

        # --- Transformer blocks with VM attention ---
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleDict({
                # Pre-norm self-attention
                "ln1": nn.LayerNorm(d_model),
                "attn": VonMisesMultiHeadAttention(
                    d_model, nhead, dropout=dropout, init_kappa=init_kappa,
                ),
                "drop1": nn.Dropout(dropout),
                # Pre-norm feedforward
                "ln2": nn.LayerNorm(d_model),
                "ffn_linear1": nn.Linear(d_model, dim_feedforward),
                "ffn_linear2": nn.Linear(dim_feedforward, d_model),
                "drop2": nn.Dropout(dropout),
            })
            self.blocks.append(block)

        # Final LayerNorm (standard in pre-norm transformers)
        self.final_ln = nn.LayerNorm(d_model)

        # --- Output head ---
        self.output_head = nn.Linear(d_model, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        # Project to d_model and add positional encoding
        h = self.input_proj(x)     # (B, T, d_model)
        h = self.pos_encoder(h)    # adds PE + dropout

        # Transformer blocks with pre-norm
        for block in self.blocks:
            # Self-attention sub-layer
            residual = h
            h_norm = block["ln1"](h)
            h = residual + block["drop1"](block["attn"](h_norm))

            # Feedforward sub-layer
            residual = h
            h_norm = block["ln2"](h)
            ffn_out = block["ffn_linear1"](h_norm)
            ffn_out = F.gelu(ffn_out)
            ffn_out = block["ffn_linear2"](ffn_out)
            h = residual + block["drop2"](ffn_out)

        # Final norm, mean pool, and project
        h = self.final_ln(h)       # (B, T, d_model)
        h_final = h.mean(dim=1)    # (B, d_model)
        return self.output_head(h_final)  # (B, output_size)

    def get_attention_layers(self) -> List[VonMisesMultiHeadAttention]:
        """Return all VM attention modules for interpretability."""
        return [block["attn"] for block in self.blocks]

    def get_attention_concentrations(self) -> List[Tensor]:
        """Return kappa matrices for all attention layers.

        Returns:
            List of tensors, each (nhead, d_head) with kappa > 0.
        """
        return [attn.get_concentrations() for attn in self.get_attention_layers()]

    def get_attention_phases(self) -> List[Tensor]:
        """Return mu matrices for all attention layers.

        Returns:
            List of tensors, each (nhead, d_head) in [0, 2pi).
        """
        return [attn.get_phases() for attn in self.get_attention_layers()]
