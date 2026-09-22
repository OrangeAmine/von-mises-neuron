"""Batched forecast fitting with validation selection including initialization.

An untrained residual head can already implement a useful persistence forecast.
Its epoch-zero checkpoint therefore competes with every trained checkpoint.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _finite(tensor: torch.Tensor, context: str) -> None:
    """Raise an explicit numerical failure instead of selecting around it."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Nonfinite {context}")


def _model_device(model: nn.Module) -> torch.device:
    """Find a model's device, including parameterless buffered models."""
    for tensor in (*model.parameters(), *model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _finite_state(model: nn.Module, context: str) -> None:
    """Check a state with one host synchronization on its common device."""
    state = model.state_dict()
    if state and not torch.stack([torch.isfinite(value).all() for value in state.values()]).all():
        for name, value in state.items():
            _finite(value, f"{context} {name}")


def batched_predict(
    model: nn.Module, x: torch.Tensor, batch_size: int = 256
) -> torch.Tensor:
    """Predict in bounded batches and return detached CPU tensors.

    Args:
        model: Fitted model on the desired inference device.
        x: Nonempty input tensor; its device need not match the model.
        batch_size: Maximum examples per inference batch.

    Returns:
        CPU predictions concatenated along the sample dimension. The incoming
        model training mode is restored, and inputs are never modified.

    Raises:
        ValueError: Inputs are empty or batch size is invalid.
        FloatingPointError: An input or prediction is nonfinite.
    """
    if batch_size < 1 or x.ndim < 1 or len(x) == 0:
        raise ValueError("Prediction requires nonempty inputs and batch_size >= 1")
    device = _model_device(model)
    was_training = model.training
    outputs = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = x[start : start + batch_size].to(device)
                _finite(batch, "prediction inputs")
                prediction = model(batch)
                if prediction.ndim < 1 or len(prediction) != len(batch):
                    raise ValueError("Prediction sample dimension does not match inputs")
                _finite(prediction, "predictions")
                outputs.append(prediction.detach().cpu())
    finally:
        model.train(was_training)
    return torch.cat(outputs, dim=0)


def _validation_mse(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int
) -> float:
    """Accumulate sample-weighted MSE without placing all validation on GPU."""
    device = _model_device(model)
    squared_error = 0.0
    elements = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            inputs = x[start : start + batch_size].to(device)
            targets = y[start : start + batch_size].to(device)
            _finite(inputs, "validation inputs")
            _finite(targets, "validation targets")
            predictions = model(inputs)
            if predictions.shape != targets.shape:
                raise ValueError("Forecast shape must exactly match target shape")
            _finite(predictions, "validation predictions")
            # Float64 summation avoids both uneven-batch weighting and avoidable
            # float32 overflow in metrics; the training objective remains MSE.
            error = (predictions.double() - targets.double()).square().sum()
            _finite(error, "validation squared error")
            squared_error += error.item()
            elements += targets.numel()
    value = squared_error / elements
    if not math.isfinite(value):
        raise FloatingPointError("Nonfinite validation loss")
    return value


def train_forecaster(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 50,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 64,
    patience: int = 15,
    gradient_clip: float = 1.0,
    validation_batch_size: int = 256,
    min_delta: float = 1e-6,
    device: torch.device | str | None = None,
    source_config: dict[str, Any] | None = None,
) -> nn.Module:
    """Fit and restore the checkpoint with the lowest validation MSE.

    Args:
        model: Forecast module to fit in place.
        X_train: Training inputs, normally (samples, lookback, features).
        y_train: Training targets, normally (samples, horizon).
        X_val: Chronologically held-out validation inputs.
        y_val: Validation targets; no test data participates in selection.
        max_epochs: Maximum training epochs; zero evaluates initialization only.
        lr: Adam learning rate.
        weight_decay: Adam weight decay.
        batch_size: Training minibatch size.
        patience: Epochs without an improvement before stopping.
        gradient_clip: Maximum gradient norm.
        validation_batch_size: Maximum examples per validation forward pass.
        min_delta: Minimum validation MSE improvement for checkpoint replacement.
        device: Training device, defaulting to CUDA when available.
        source_config: Optional JSON-compatible caller provenance copied to history.

    Returns:
        The same model, in evaluation mode at its best checkpoint, with a
        JSON-compatible ``training_history``. Validation losses include epoch
        zero, so ``validation_loss[best_epoch]`` is the selected MSE.

    Raises:
        ValueError: Data/configuration shapes or ranges are invalid.
        FloatingPointError: Any checked input, output, loss, gradient or state
            is nonfinite. Failure history remains attached; no checkpoint is
            silently restored to disguise a failed training run.
    """
    if max_epochs < 0 or min(batch_size, validation_batch_size, patience) < 1:
        raise ValueError("Epoch count must be nonnegative; batch sizes and patience positive")
    if any(not math.isfinite(v) for v in (lr, weight_decay, gradient_clip, min_delta)):
        raise ValueError("Optimizer and selection settings must be finite")
    if lr <= 0 or weight_decay < 0 or gradient_clip <= 0 or min_delta < 0:
        raise ValueError("Invalid optimizer or selection setting")
    for label, x, y in (("training", X_train, y_train), ("validation", X_val, y_val)):
        if x.ndim < 2 or y.ndim != 2 or len(x) == 0 or y.numel() == 0 or len(x) != len(y):
            raise ValueError(f"Invalid {label} input/target sample dimensions")
        if not x.is_floating_point() or not y.is_floating_point():
            raise ValueError(f"{label.capitalize()} inputs and targets must be floating point")
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(selected_device)
    history: dict[str, Any] = {
        "status": "running", "epochs_completed": 0, "best_epoch": None,
        "best_val_loss": None, "validation_epochs": [], "validation_loss": [],
        "training_loss": [], "learning_rate": [], "max_epochs": max_epochs,
        "lr": lr, "weight_decay": weight_decay, "batch_size": batch_size,
        "validation_batch_size": validation_batch_size, "patience": patience,
        "gradient_clip": gradient_clip, "min_delta": min_delta,
        "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 5, "min_lr": 1e-5},
        "checkpoint_selection": "validation_mse_including_epoch_zero",
        "device": str(selected_device), "source_config": deepcopy(source_config or {}),
    }
    model.training_history = history
    current_epoch = 0
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
        )
        loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        _finite_state(model, "initial model state")
        best_loss = _validation_mse(model, X_val, y_val, validation_batch_size)
        history["validation_epochs"].append(0)
        history["validation_loss"].append(best_loss)
        history.update(best_epoch=0, best_val_loss=best_loss)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        patience_counter = 0
        for current_epoch in range(1, max_epochs + 1):
            model.train()
            train_error, train_elements = 0.0, 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for inputs, targets in loader:
                inputs, targets = inputs.to(selected_device), targets.to(selected_device)
                _finite(inputs, "training inputs")
                _finite(targets, "training targets")
                optimizer.zero_grad(set_to_none=True)
                prediction = model(inputs)
                if prediction.shape != targets.shape:
                    raise ValueError("Forecast shape must exactly match target shape")
                _finite(prediction, "training predictions")
                loss = nn.functional.mse_loss(prediction, targets)
                _finite(loss, "training loss")
                loss.backward()
                norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                _finite(norm, "gradient norm")
                optimizer.step()
                train_error += loss.item() * targets.numel()
                train_elements += targets.numel()
            # Check even unused parameters/buffers before accepting an epoch;
            # doing this once per epoch avoids a GPU sync per parameter/batch.
            _finite_state(model, "model state")
            val_loss = _validation_mse(model, X_val, y_val, validation_batch_size)
            history["training_loss"].append(train_error / train_elements)
            history["learning_rate"].append(epoch_lr)
            history["validation_epochs"].append(current_epoch)
            history["validation_loss"].append(val_loss)
            history["epochs_completed"] = current_epoch
            scheduler.step(val_loss)
            if val_loss < best_loss - min_delta:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                history.update(best_epoch=current_epoch, best_val_loss=best_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        model.load_state_dict(best_state)
        model.eval()
        history.update(status="completed", stopped_early=history["epochs_completed"] < max_epochs)
    except Exception as exc:
        history.update(status="failed", failed_epoch=current_epoch, failure=f"{type(exc).__name__}: {exc}")
        raise
    return model
