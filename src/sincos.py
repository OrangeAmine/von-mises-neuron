"""Architecture-matched Cartesian control for the base Von Mises forecaster.

The identity ``kappa*cos(x+mu) = a*cos(x) + b*sin(x)`` holds for
``a=kappa*cos(mu)`` and ``b=-kappa*sin(mu)``. Each layer therefore has the
same number of trainable coefficients as a VM layer, but optimizes Cartesian
coefficients directly rather than phases and unconstrained softplus parameters.
The function classes agree except that exactly zero amplitude is attainable
with finite Cartesian parameters, while strictly positive softplus amplitude
only approaches zero. A stacked encoder remains nonlinear in its original
inputs; this identity does not make the whole forecaster linear.

This control retains the VM model's treatment of every sensor as an angle.
It tests parameterization, not whether that periodic inductive bias is suitable
for a particular sensor. Periodicity holds at each layer's input, and end to
end in evaluation mode because there is no raw-input residual. Neither model
is a normalized density. Optional exponential outputs can overflow.

For paired experiments, convert an untrained VM model with ``from_von_mises``
and reset training RNG identically before each fit. Use the same data, optimizer
settings, batches, stopping rule, and budget. Matching initial functions does
not match optimizer trajectories or imply equal computational cost: optimizer
updates are coordinate-dependent and the arithmetic implementations differ.
"""

import copy
from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.von_mises_neuron import VonMisesForecaster, VonMisesLinear


class SinCosLinear(nn.Module):
    """Learn a Cartesian linear combination of per-input sine/cosine features.

    Args:
        in_features: Number of angular input features.
        out_features: Number of output features.
        bias: Whether to add an output bias.
        log_output: Return the trigonometric sum, or exponentiate before bias.
        init_kappa: Initial coefficient amplitude, following VM initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        log_output: bool = True,
        init_kappa: float = 1.0,
    ) -> None:
        super().__init__()
        reference = VonMisesLinear(
            in_features, out_features, bias, log_output, init_kappa
        )
        self._copy_from_von_mises(reference)

    def _copy_from_von_mises(self, reference: VonMisesLinear) -> None:
        """Populate independent parameters without consuming random numbers."""
        self.in_features = reference.in_features
        self.out_features = reference.out_features
        self.log_output = reference.log_output
        with torch.no_grad():
            kappa = F.softplus(reference.log_kappa)
            self.weight_cos = nn.Parameter(kappa * reference.mu.cos())
            self.weight_sin = nn.Parameter(-kappa * reference.mu.sin())
        if reference.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(reference.bias.detach().clone())
        self.train(reference.training)

    @classmethod
    def from_von_mises(cls, reference: VonMisesLinear) -> "SinCosLinear":
        """Copy a VM layer's function, dtype, device, and mode without RNG draws.

        Args:
            reference: Source layer; its parameters are neither changed nor shared.

        Returns:
            Independently trainable Cartesian layer with equivalent outputs.
        """
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._copy_from_von_mises(reference)
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(..., in_features)`` angles to ``(..., out_features)`` outputs."""
        z = F.linear(x.cos(), self.weight_cos) + F.linear(x.sin(), self.weight_sin)
        out = z if self.log_output else z.exp()
        return out if self.bias is None else out + self.bias

    def extra_repr(self) -> str:
        """Describe the dimensions and output mode."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, log_output={self.log_output}"
        )


class SinCosForecaster(nn.Module):
    """Base VM architecture with only the polar coefficients reparameterized.

    Arguments and defaults match ``VonMisesForecaster``. Construction first
    initializes that architecture, then converts it, so resetting the same seed
    before both constructors also gives matched initial functions. The explicit
    ``from_von_mises`` factory is preferred for an already-created reference.

    Args:
        input_size: Input features per timestep.
        hidden_size: Encoder width and LSTM hidden size.
        num_vm_layers: Number of sine/cosine blocks; name retained for matching.
        dropout: Encoder dropout probability.
        output_size: Forecast horizon.
        init_kappa: Initial coefficient amplitude, following VM initialization.
        log_output: Use trigonometric sums instead of exponential outputs.
        use_layernorm: Include the reference LayerNorm after each encoder layer.
        use_lstm: Use the reference LSTM, or its mean-pooling ablation.
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
        reference = VonMisesForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_vm_layers=num_vm_layers,
            dropout=dropout,
            output_size=output_size,
            init_kappa=init_kappa,
            log_output=log_output,
            use_layernorm=use_layernorm,
            use_lstm=use_lstm,
        )
        self._copy_from_von_mises(reference)

    def _copy_from_von_mises(self, reference: VonMisesForecaster) -> None:
        """Copy the architecture, replacing only its VM layers."""
        self.input_size = reference.input_size
        self.hidden_size = reference.hidden_size
        self.num_vm_layers = reference.num_vm_layers
        self.output_size = reference.output_size
        self.sincos_encoder = nn.Sequential(*[
            SinCosLinear.from_von_mises(layer)
            if isinstance(layer, VonMisesLinear) else copy.deepcopy(layer)
            for layer in reference.vm_encoder
        ])
        self.lstm = copy.deepcopy(reference.lstm)
        if self.lstm is not None:
            # Deep-copying CUDA RNN weights loses cuDNN's packed storage.
            self.lstm.flatten_parameters()
        self.output_head = copy.deepcopy(reference.output_head)
        self.train(reference.training)

    @classmethod
    def from_von_mises(cls, reference: VonMisesForecaster) -> "SinCosForecaster":
        """Copy reference weights and structure with no shared tensors or RNG draws.

        Args:
            reference: Base VM forecaster to reparameterize.

        Returns:
            A forecaster with matching dtype, device, mode and initial function.
        """
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._copy_from_von_mises(reference)
        return result

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(batch, lookback, features)`` to ``(batch, horizon)``."""
        h = self.sincos_encoder(x)
        if self.lstm is None:
            h_final = h.mean(dim=1)
        else:
            _, (h_n, _) = self.lstm(h)
            h_final = h_n.squeeze(0)
        return self.output_head(h_final)

    def get_sincos_layers(self) -> List[SinCosLinear]:
        """Return the Cartesian feature layers in encoder order."""
        return [layer for layer in self.sincos_encoder if isinstance(layer, SinCosLinear)]
