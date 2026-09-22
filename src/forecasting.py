"""Persistence residual forecasts for real-valued sensor time series."""

from numbers import Integral

import torch
from torch import Tensor, nn


class PersistenceResidualForecaster(nn.Module):
    """Predict changes relative to the last observed target value.

    Each sensor channel is centered by its own last observation before the
    supplied forecaster sees it. Trailing calendar angles are passed through
    unchanged. The body predicts a target increment, and the target's last
    normalized observation is added at every forecast horizon. Inputs and
    targets must use the same target normalization and units.

    Constant shifts to any sensor's entire history leave the increment
    unchanged; shifting the target history shifts the forecast by that amount.
    This is level translation equivariance, not an angular periodicity claim.
    With dropout, compare deterministic evaluation or identical random draws.

    Args:
        body: Existing forecaster with ``output_size`` and a Linear
            ``output_head``. VM, SinCos and LSTM bodies expose ``input_size``;
            TemporalVM instead exposes ``sensor_size`` and ``time_periods``.
            The module is owned directly, without copying its parameters.
        sensor_size: Number of sensor columns at the beginning of the input.
        target_index: Target column index, within those sensor columns.
        zero_init_output: Reset the body's existing output-head weight and
            bias to zero, starting at exact persistence for finite body
            activations. Set False to preserve an already learned increment
            head. No extra learnable parameters are introduced.

    Zero initialization initially blocks gradients to upstream body weights.
    The head learns on the first optimizer step; upstream learning can then
    begin. The centering operation remains differentiable, including its
    dependence on the last observation.
    """

    def __init__(
        self,
        body: nn.Module,
        sensor_size: int,
        target_index: int,
        *,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(body, nn.Module):
            raise TypeError("body must be a torch.nn.Module")
        input_size = getattr(body, "input_size", None)
        if input_size is None and hasattr(body, "time_periods"):
            input_size = body.sensor_size + len(body.time_periods)
        output_size = getattr(body, "output_size", None)
        for name, value in (
            ("body input_size", input_size),
            ("body output_size", output_size),
            ("sensor_size", sensor_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if sensor_size > input_size:
            raise ValueError("sensor_size cannot exceed body input_size")
        if hasattr(body, "sensor_size") and body.sensor_size != sensor_size:
            raise ValueError("sensor_size must match the temporal body's sensor_size")
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, Integral)
            or not 0 <= target_index < sensor_size
        ):
            raise ValueError("target_index must index one of the sensor columns")
        if not isinstance(zero_init_output, bool):
            raise TypeError("zero_init_output must be a bool")
        output_head = getattr(body, "output_head", None)
        if not isinstance(output_head, nn.Linear):
            raise TypeError("body.output_head must be a torch.nn.Linear")
        if output_head.out_features != output_size:
            raise ValueError("output_head width must match body.output_size")

        self.body = body
        self.input_size = int(input_size)
        self.sensor_size = int(sensor_size)
        self.target_index = int(target_index)
        self.output_size = int(output_size)
        self.zero_init_output = zero_init_output
        if zero_init_output:
            nn.init.zeros_(output_head.weight)
            if output_head.bias is not None:
                nn.init.zeros_(output_head.bias)
        self.train(body.training)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(batch, lookback, features)`` to ``(batch, horizon)``.

        Args:
            x: Floating sensor histories followed by optional raw time angles.

        Returns:
            Forecasts in the normalized target's units.

        Raises:
            ValueError: Input or body-output dimensions violate the contract.
            TypeError: Inputs are not floating tensors.
        """
        if not isinstance(x, Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating-point torch.Tensor")
        if x.ndim != 3 or x.shape[-1] != self.input_size:
            raise ValueError(
                f"x must have shape (batch, lookback, {self.input_size})"
            )
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch and lookback must be nonempty")
        sensors = x[..., :self.sensor_size]
        # Preserve clock phases while removing only sensor level offsets.
        centered = sensors - sensors[:, -1:, :]
        body_input = torch.cat((centered, x[..., self.sensor_size:]), dim=-1)
        increment = self.body(body_input)
        if increment.shape != (x.shape[0], self.output_size):
            raise ValueError("body must return shape (batch, output_size)")
        last_target = x[:, -1, self.target_index:self.target_index + 1]
        return last_target + increment
