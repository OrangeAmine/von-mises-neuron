"""Von Mises (VM) neuron for natively handling periodic/circular data.

The standard neuron applies a weight *inside* a nonlinearity:

    s = f(W₁·θ + W₀)

When f = cos or exp∘cos, multiplying θ by W₁ changes the periodicity —
the neuron no longer respects the original 2π-period of the input.

The VM neuron fixes this by placing the weight *outside* the cosine:

    s = exp(κ · cos(θ + μ))                               ... (1)

where:
    κ > 0  = concentration (controls peak sharpness, analogous to 1/σ²)
    μ ∈ ℝ  = phase shift (preferred direction)
    θ      = input (treated as an angle in radians)

This is exactly the von Mises density (circular Gaussian) up to a
normalising constant.  The output is strictly 2π-periodic in θ.

Multi-dimensional extension (VM linear layer):

    output_j = exp( Σᵢ κⱼᵢ · cos(θᵢ + μⱼᵢ) ) + bⱼ       ... (2)

This is a product of independent von Mises distributions — the log is
a sum of cosines, each preserving 2π-periodicity in its input dimension.

Gradient properties (for Eq. 2 with z_j = Σ κ cos(θ + μ)):
    ∂s/∂μⱼᵢ  = −s · κⱼᵢ · sin(θᵢ + μⱼᵢ)   — vanishes at peaks of cos
    ∂s/∂κⱼᵢ  =  s · cos(θᵢ + μⱼᵢ)          — proportional to cos (bounded)
    ∂s/∂θᵢ   = −s · κⱼᵢ · sin(θᵢ + μⱼᵢ)   — same form as ∂s/∂μ

The vanishing gradient at cos peaks is *desirable*: once a neuron's phase
aligns with the input, it stops drifting.  The concentration κ still
receives a gradient proportional to cos, so it can always sharpen or
soften the peak.

Use cases:
    - Wind direction / ocean current forecasting
    - Protein dihedral angle prediction (φ/ψ Ramachandran angles)
    - Robotics joint angle control
    - Time-of-day / seasonal features (hours, months encoded as angles)
    - Phase estimation in signal processing
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class VonMisesLinear(nn.Module):
    """Linear-layer equivalent using Von Mises neurons.

    Replaces the standard affine transform z = Wx + b with:

        z_j = Σᵢ κⱼᵢ · cos(xᵢ + μⱼᵢ)        (log-space)
    or  s_j = exp(z_j) + bⱼ                    (exp-space, default)

    Each neuron j has a preferred direction μⱼᵢ for each input i and a
    concentration κⱼᵢ that controls how sharply it responds to inputs
    near that direction.

    Args:
        in_features: Number of input features.
        out_features: Number of output neurons.
        bias: If True, include output bias (default True).
        log_output: If True, output the log-space z instead of exp(z).
            Useful for stacking VM layers — avoids exp explosion and
            keeps values in a range suitable as input to the next cos().
        init_kappa: Initial concentration value (default 1.0).
            Higher = sharper peaks, lower = broader response.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        log_output: bool = False,
        init_kappa: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_output = log_output

        # μ: phase shifts (preferred directions), one per (neuron, input) pair
        self.mu = nn.Parameter(torch.empty(out_features, in_features))

        # κ stored in unconstrained log-space; softplus ensures κ > 0
        self.log_kappa = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Cache target for reset_parameters
        self._init_kappa_target = init_kappa
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize μ uniformly in [0, 2π) and κ near init_kappa.

        Uniform μ gives full angular coverage at initialisation so that
        different neurons "look at" different phases from the start.
        """
        nn.init.uniform_(self.mu, 0, 2 * math.pi)
        # softplus⁻¹(κ) = log(exp(κ) − 1); guard against κ ≈ 0
        if self._init_kappa_target > 0.05:
            inv = math.log(math.exp(self._init_kappa_target) - 1)
        else:
            inv = -3.0  # softplus(-3) ≈ 0.049
        nn.init.constant_(self.log_kappa, inv)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Von Mises activation.

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        # Eq. κ > 0 via softplus: κ = log(1 + exp(raw))
        kappa = nn.functional.softplus(self.log_kappa)  # (out, in)

        # Broadcast x over the output dimension:
        # x: (..., in) → (..., 1, in)  |  mu: (out, in) → broadcast
        # cos(xᵢ + μⱼᵢ) for every (neuron j, input i) pair
        cos_term = torch.cos(x.unsqueeze(-2) + self.mu)  # (..., out, in)

        # z_j = Σᵢ κⱼᵢ · cos(xᵢ + μⱼᵢ)  —  the log-space VM response
        z = (kappa * cos_term).sum(dim=-1)  # (..., out)

        if self.log_output:
            # Log-space output: suitable for feeding into the next VM layer
            # because cos() accepts any real-valued input (it wraps by 2π)
            out = z
        else:
            # Exp-space output: the actual von Mises response
            # s_j = exp(z_j) — always positive, peaked where inputs align
            out = torch.exp(z)

        if self.bias is not None:
            out = out + self.bias

        return out

    def get_concentrations(self) -> Tensor:
        """Return the positive concentration matrix κ.

        Returns:
            Tensor of shape (out_features, in_features) with κ > 0.
        """
        return nn.functional.softplus(self.log_kappa).detach()

    def get_phases(self) -> Tensor:
        """Return phase shifts μ wrapped to [0, 2π).

        Returns:
            Tensor of shape (out_features, in_features) in [0, 2π).
        """
        return (self.mu.detach() % (2 * math.pi))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"log_output={self.log_output}"
        )


class VonMisesForecaster(nn.Module):
    """Direct multi-step forecaster with VM feature encoding + temporal backbone.

    Default architecture::

        Input (B, T, F)
          → [VonMisesLinear(log) → LayerNorm → Dropout] × N   (VM encoder)
          → LSTM temporal backbone                              (sequence model)
          → Linear head from final hidden state                 (projection)

    The VM encoder processes each timestep independently, extracting periodic
    feature representations that are 2π-equivariant in the input.  The LSTM
    backbone captures temporal dependencies across the encoded timesteps,
    and the linear head maps the final hidden state to the forecast horizon.

    This separation of concerns is intentional: VM layers handle periodic
    feature encoding (what they're designed for), while the LSTM handles
    sequential dynamics (what it's designed for).

    Ablation variants (controlled via optional arguments):

    - ``log_output=False``: VM layers output exp(z) instead of z between
      layers.  Expected to cause instability in deep stacks because
      exp(exp(z)) grows superexponentially.
    - ``use_layernorm=False``: Remove LayerNorm between VM layers.
      Without it, neurons with high κ dominate downstream activations.
    - ``use_lstm=False``: Replace the LSTM with mean pooling over time.
      Isolates the VM encoder's contribution from temporal modeling.

    Args:
        input_size: Number of input features per timestep.
        hidden_size: Number of VM neurons per layer / LSTM hidden size.
        num_vm_layers: Number of stacked VM layers (default 2).
        dropout: Dropout rate between VM layers and in LSTM (default 0.0).
        output_size: Forecast horizon (number of output steps).
        init_kappa: Initial concentration for all VM layers.
        log_output: If True (default), VM layers output log-space z for
            inter-layer communication.  If False, output exp(z).
        use_layernorm: If True (default), include LayerNorm after each
            VM layer to normalize scale across neurons.
        use_lstm: If True (default), use LSTM for temporal modeling.
            If False, use mean pooling over the time dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_vm_layers: int = 2,
        dropout: float = 0.0,
        output_size: int = 1,
        init_kappa: float = 1.0,
        log_output: bool = True,
        use_layernorm: bool = True,
        use_lstm: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_vm_layers = num_vm_layers
        self.output_size = output_size
        self._log_output = log_output
        self._use_layernorm = use_layernorm
        self._use_lstm = use_lstm

        # --- VM feature encoder ---
        # When log_output=True (default), log-space output avoids exp
        # explosion between layers and keeps values in a range where the
        # next cos() uses its full curvature.
        layers: List[nn.Module] = []
        in_dim = input_size
        for _ in range(num_vm_layers):
            layers.append(
                VonMisesLinear(
                    in_dim,
                    hidden_size,
                    bias=True,
                    log_output=log_output,
                    init_kappa=init_kappa,
                )
            )
            if use_layernorm:
                # LayerNorm compensates for scale inconsistencies across neurons
                # (neurons with higher κ produce larger-magnitude outputs)
                layers.append(nn.LayerNorm(hidden_size))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size

        self.vm_encoder = nn.Sequential(*layers)

        # --- Temporal backbone ---
        if use_lstm:
            # Single-layer LSTM to capture sequential dynamics from VM features.
            # The VM encoder extracts *what* periodic patterns exist at each
            # timestep; the LSTM learns *how* those patterns evolve over time.
            self.lstm: Optional[nn.LSTM] = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,  # single layer, dropout handled by VM encoder
            )
        else:
            # Mean pooling over time — isolates VM encoder contribution
            # by removing the LSTM's temporal modeling capability
            self.lstm = None

        # --- Output head ---
        self.output_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, input_size).

        Returns:
            Forecast tensor, shape (batch, output_size).
        """
        # VM encode: extract periodic features per timestep
        # (B, T, F) → (B, T, H)
        h = self.vm_encoder(x)

        if self.lstm is not None:
            # LSTM: capture temporal dependencies across timesteps
            # (B, T, H) → h_n: (1, B, H)
            _, (h_n, _) = self.lstm(h)
            # Use final hidden state: (1, B, H) → (B, H)
            h_final = h_n.squeeze(0)
        else:
            # Mean pool over time dimension: (B, T, H) → (B, H)
            h_final = h.mean(dim=1)

        # Linear projection to forecast horizon: (B, H) → (B, output_size)
        return self.output_head(h_final)

    def get_vm_layers(self) -> List[VonMisesLinear]:
        """Return list of VonMisesLinear layers for interpretability.

        Useful for inspecting learned concentrations and phase shifts:
            for vm in model.get_vm_layers():
                print(vm.get_concentrations())
                print(vm.get_phases())
        """
        return [m for m in self.vm_encoder if isinstance(m, VonMisesLinear)]

    def get_layer_concentrations(self) -> List[Tensor]:
        """Return concentration matrices for all VM layers.

        Returns:
            List of tensors, each (out_features, in_features) with κ > 0.
        """
        return [vm.get_concentrations() for vm in self.get_vm_layers()]

    def get_layer_phases(self) -> List[Tensor]:
        """Return phase shift matrices for all VM layers.

        Returns:
            List of tensors, each (out_features, in_features) in [0, 2π).
        """
        return [vm.get_phases() for vm in self.get_vm_layers()]
