"""Bounded paired neural and ridge fitting for the VM-N04 protocol."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA before reading wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_deadline(deadline: float) -> None:
    """Stop at the prospective stage deadline without silently pruning fits."""
    if time.perf_counter() >= deadline:
        raise TimeoutError("VM-N04 prospective stage runtime cap reached")


def fit_neural(model: nn.Module, train: Any, validation: Any,
               settings: dict, lr: float, seed: int, device: torch.device,
               deadline: float) -> dict:
    """Fit fixed epochs and restore the validation-selected checkpoint.

    All arrays must already follow the split/availability contract. RNG reset
    after construction pairs minibatches regardless of parameterization.
    """
    synchronize(device)
    started = time.perf_counter()
    model.to(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    tx, ta, ty = [torch.as_tensor(a, dtype=torch.float32, device=device)
                  for a in (train.sensor, train.future_angles, train.y)]
    vx, va, vy = [torch.as_tensor(a, dtype=torch.float32, device=device)
                  for a in (validation.sensor, validation.future_angles, validation.y)]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=settings["weight_decay"])
    best_loss, best_epoch, best_state = float("inf"), -1, None
    history = []
    generator = torch.Generator(device="cpu").manual_seed(seed + 100000)
    for epoch in range(settings["epochs"] + 1):
        check_deadline(deadline)
        if epoch:
            model.train()
            indices = torch.randperm(len(tx), generator=generator).to(device)
            for batch in indices.split(settings["batch_size"]):
                check_deadline(deadline)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tx[batch], ta[batch])
                if prediction.shape != ty[batch].shape:
                    raise ValueError("Forecast/target shape mismatch")
                loss = (prediction - ty[batch]).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                         error_if_nonfinite=True)
                optimizer.step()
        if epoch % settings["validation_every"] == 0 or epoch == settings["epochs"]:
            model.eval()
            with torch.no_grad():
                value = float((model(vx, va).double() - vy.double()).square().mean())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite validation loss")
            history.append({"epoch": epoch, "validation_mse": value})
            if value < best_loss:
                best_loss, best_epoch = value, epoch
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    return {"training_seconds": time.perf_counter() - started, "best_epoch": best_epoch,
            "validation_mse": best_loss, "history": history,
            "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def predict_neural(model: nn.Module, batch: Any, device: torch.device,
                   angle_offset: float = 0.0) -> tuple[np.ndarray, float]:
    """Predict normalized targets, including transfer time, in bounded batches."""
    synchronize(device)
    started = time.perf_counter()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(batch.sensor), 1024):
            x = torch.as_tensor(batch.sensor[start:start + 1024], device=device)
            a = torch.as_tensor(batch.future_angles[start:start + 1024], device=device)
            predictions.append(model(x, a + angle_offset).cpu().numpy())
    result = np.concatenate(predictions)
    if result.shape != batch.y.shape or not np.isfinite(result).all():
        raise FloatingPointError("Invalid inference output")
    synchronize(device)
    return result, time.perf_counter() - started


def ridge_design(sensor: np.ndarray) -> np.ndarray:
    """Flatten the same sensor history; the persistence residual is separate."""
    return sensor.reshape(len(sensor), -1).astype(np.float64)


def fit_ridge(train: Any, validation: Any, penalties: list[float]) -> dict:
    """Select mean-objective positive ridge penalty on validation labels only."""
    started = time.perf_counter()
    x, vx = ridge_design(train.sensor), ridge_design(validation.sensor)
    y = train.y.astype(np.float64) - train.sensor[:, -1, 0, None]
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    gram, rhs = xc.T @ xc, xc.T @ yc
    candidates, states = [], []
    for penalty in penalties:
        w = np.linalg.solve(gram + len(x) * penalty * np.eye(x.shape[1]), rhs)
        b = ym - xm @ w
        pred = validation.sensor[:, -1, 0, None] + vx @ w + b
        mse = float(np.mean((pred - validation.y) ** 2))
        if not np.isfinite(mse):
            raise FloatingPointError("Nonfinite ridge validation")
        candidates.append({"penalty": penalty, "validation_mse": mse})
        states.append((w, b))
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["validation_mse"])
    w, b = states[selected]
    return {"weight": w, "bias": b, "selected_penalty": penalties[selected],
            "candidates": candidates, "parameters": int(w.size + b.size),
            "training_seconds": time.perf_counter() - started}


def error_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Return finite ppm errors, or explicit nulls for empty diagnostic strata."""
    if prediction.shape != target.shape:
        raise ValueError("Metric shapes must agree exactly")
    if not prediction.size:
        return {"n": 0, "rmse": None, "mae": None}
    error = prediction.astype(np.float64) - target.astype(np.float64)
    if not np.isfinite(error).all():
        raise FloatingPointError("Nonfinite error")
    return {"n": int(error.size), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error)))}
