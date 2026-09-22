"""Causal controls and frozen practical forecasting selection rules.

Metrics expect original-unit power forecasts (kW). Ridge inputs and targets
remain in the training-only normalization used by their caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import Ridge


def _array(value: np.ndarray, name: str, ndim: int) -> np.ndarray:
    """Validate numerical arrays without changing their source contents."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Nonfinite {name}")
    return result


def forecast_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Calculate pooled and per-horizon errors in original power units.

    Args:
        prediction: Forecasts of shape (origins, horizon), in kW.
        target: Matching observed targets, in kW.

    Returns:
        RMSE/MAE in kW, pooled R2 (None for a constant target), WAPE percent
        (None when absolute target sum is zero), MAPE percent with denominator
        ``max(abs(target), 0.1 kW)``, per-horizon RMSE, SSE, and target count.
        The MAPE clamp is fixed to avoid unbounded percentage errors near zero.
    """
    prediction = _array(prediction, "prediction", 2)
    target = _array(target, "target", 2)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match exactly")
    with np.errstate(over="raise", invalid="raise"):
        error = prediction - target
        squared = error ** 2
        absolute = np.abs(error)
        sse = float(squared.sum())
        total_variation = float(((target - target.mean()) ** 2).sum())
        total_absolute_target = float(np.abs(target).sum())
        return {
            "rmse": float(np.sqrt(squared.mean())),
            "mae": float(absolute.mean()),
            "r2": None if total_variation == 0 else float(1 - sse / total_variation),
            "wape": None if total_absolute_target == 0 else float(100 * absolute.sum() / total_absolute_target),
            "mape": float(100 * (absolute / np.maximum(np.abs(target), 0.1)).mean()),
            "mape_denominator_floor_kw": 0.1,
            "per_horizon_rmse": np.sqrt(squared.mean(axis=0)).tolist(),
            "sse": sse,
            "n_targets": int(target.size),
        }


def _inputs(X: np.ndarray, target_index: int, n_time_features: int = 0) -> np.ndarray:
    """Check input ordering and target membership in sensor columns."""
    x = _array(X, "inputs", 3)
    if not isinstance(n_time_features, (int, np.integer)) or not 0 <= n_time_features < x.shape[2]:
        raise ValueError("Time feature count must leave at least one sensor")
    if not isinstance(target_index, (int, np.integer)) or not 0 <= target_index < x.shape[2] - n_time_features:
        raise ValueError("target_index must identify a sensor column")
    return x


def simple_predictions(
    X: np.ndarray, target_index: int, horizon: int
) -> dict[str, np.ndarray]:
    """Predict solely from observed normalized target history.

    Args:
        X: Inputs (origins, lookback, sensors followed by optional angles).
        target_index: Target's sensor-column index.
        horizon: Positive forecast horizon no greater than 24 hours.

    Returns:
        Persistence, Seasonal24 and Seasonal168 predictions in normalized
        target units. At horizon step h (one-based), season s uses X[-s+h-1].

    Raises:
        ValueError: Fewer than 168 observed hours or a horizon beyond the
            shortest season would require an unavailable observation.
    """
    x = _inputs(X, target_index)
    if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= 24:
        raise ValueError("horizon must be an integer in [1, 24] for causal seasonal controls")
    if x.shape[1] < 168:
        raise ValueError("Seasonal168 requires at least 168 observed hours")
    result = {"Persistence": np.repeat(x[:, -1, target_index, None], horizon, axis=1)}
    for season in (24, 168):
        result[f"Seasonal{season}"] = x[:, x.shape[1] - season + np.arange(horizon), target_index].copy()
    return result


def _ridge_features(X: np.ndarray, target_index: int, n_time_features: int) -> np.ndarray:
    """Center each sensor history on its origin level; encode origin angles."""
    x = _inputs(X, target_index, n_time_features)
    n_sensors = x.shape[2] - n_time_features
    sensors = x[:, :, :n_sensors]
    centered = (sensors - sensors[:, -1:, :]).reshape(len(x), -1)
    if n_time_features == 0:
        return centered
    angles = x[:, -1, n_sensors:]
    return np.concatenate((centered, np.sin(angles), np.cos(angles)), axis=1)


@dataclass
class RidgeResidual:
    """A fitted residual ridge, or explicit persistence validation fallback."""

    estimator: Ridge | None
    target_index: int
    n_time_features: int
    horizon: int
    input_shape: tuple[int, int]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return normalized target levels without future observations."""
        x = _inputs(X, self.target_index, self.n_time_features)
        if x.shape[1:] != self.input_shape:
            raise ValueError("Prediction lookback/features must match ridge training inputs")
        base = np.repeat(x[:, -1, self.target_index, None], self.horizon, axis=1)
        if self.estimator is not None:
            correction = self.estimator.predict(_ridge_features(x, self.target_index, self.n_time_features))
            base += correction.reshape(len(x), self.horizon)
        if not np.isfinite(base).all():
            raise FloatingPointError("Nonfinite ridge predictions")
        return base


def fit_ridge_residual(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_index: int,
    n_time_features: int = 2,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
) -> tuple[RidgeResidual, dict[str, Any]]:
    """Fit fixed ridge penalties on training data and select by validation MSE.

    Args:
        X_train: Normalized training sensor histories and raw trailing angles.
        y_train: Normalized training target levels (origins, horizon).
        X_val: Chronologically held-out validation inputs.
        y_val: Validation target levels, in the same train-fitted normalization.
        target_index: Target sensor index.
        n_time_features: Number of trailing raw angular features.
        alphas: Fixed positive ridge penalties; no test data is accepted.

    Returns:
        Fitted predictor and JSON-compatible validation-selection metadata.
        Persistence competes explicitly and wins numerical ties. Ridge models
        fit an intercept on training residuals only; selected coefficients stay
        available through ``predictor.estimator.coef_``.
    """
    x_train = _inputs(X_train, target_index, n_time_features)
    x_val = _inputs(X_val, target_index, n_time_features)
    train_targets = _array(y_train, "training targets", 2)
    val_targets = _array(y_val, "validation targets", 2)
    if x_train.shape[1:] != x_val.shape[1:]:
        raise ValueError("Training and validation feature shapes must match")
    if len(x_train) != len(train_targets) or len(x_val) != len(val_targets):
        raise ValueError("Input and target sample counts must match")
    if train_targets.shape[1] != val_targets.shape[1]:
        raise ValueError("Training and validation horizons must match")
    penalties = [float(alpha) for alpha in alphas]
    if not penalties or any(not np.isfinite(a) or a <= 0 for a in penalties):
        raise ValueError("Ridge penalties must be a nonempty sequence of finite positive values")
    train_features = _ridge_features(x_train, target_index, n_time_features)
    val_features = _ridge_features(x_val, target_index, n_time_features)
    residual_targets = train_targets - x_train[:, -1, target_index, None]
    base = np.repeat(x_val[:, -1, target_index, None], val_targets.shape[1], axis=1)
    best_score = float(np.mean((base - val_targets) ** 2))
    if not np.isfinite(best_score):
        raise FloatingPointError("Nonfinite persistence validation loss")
    scores = [{"candidate": "Persistence", "alpha": None, "validation_mse": best_score}]
    selected: Ridge | None = None
    selected_alpha = None
    for alpha in penalties:
        estimator = Ridge(alpha=alpha, fit_intercept=True)
        estimator.fit(train_features, residual_targets)
        prediction = base + estimator.predict(val_features).reshape(val_targets.shape)
        score = float(np.mean((prediction - val_targets) ** 2))
        if not np.isfinite(score) or not np.isfinite(estimator.coef_).all() or not np.isfinite(estimator.intercept_).all():
            raise FloatingPointError(f"Nonfinite ridge fit/validation at alpha={alpha}")
        scores.append({"candidate": "RidgeResidual", "alpha": alpha, "validation_mse": score})
        if score < best_score - 1e-12:
            selected, selected_alpha, best_score = estimator, alpha, score
    predictor = RidgeResidual(selected, target_index, n_time_features,
                              train_targets.shape[1], x_train.shape[1:])
    metadata = {
        "selected_alpha": selected_alpha,
        "selected_candidate": "Persistence" if selected is None else "RidgeResidual",
        "persistence_selected": selected is None,
        "best_val_loss": best_score,
        "validation_mse": best_score,
        "validation_scores": scores,
        "alphas": penalties,
        "fit_intercept": True,
        "n_features": int(train_features.shape[1]),
        "n_training_origins": len(x_train),
        "n_validation_origins": len(x_val),
        "selection": "validation_mse_with_persistence_competitor",
        "passed": True,
    }
    return predictor, metadata


def select_candidate(
    fold_metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    simple_families: Sequence[str] = ("Persistence", "Seasonal24", "Seasonal168"),
    family_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply frozen development gates to common-origin, per-family scores.

    Args:
        fold_metrics: Fold -> family -> metrics from ``forecast_metrics``.
            Neural family entries must describe equal-weight seed ensembles.
        simple_families: Fixed simple controls, also ordered by simplicity.
        family_order: Optional learned-family tie order; remaining families use
            alphabetical order. Simple controls always win numerical ties.

    Returns:
        Selected family, best simple family, pooled aggregate scores, and
        every learned family's checks. A learned family qualifies only at
        least 5% below best-simple pooled RMSE and no more than 10% above each
        fold's best-simple RMSE. Among qualifying families choose lowest pooled
        RMSE, breaking ties by the declared order. Otherwise select best simple.
    """
    if not fold_metrics or not simple_families or len(set(simple_families)) != len(simple_families):
        raise ValueError("Nonempty folds and distinct simple families are required")
    first = next(iter(fold_metrics.values()))
    families = set(first)
    if not set(simple_families).issubset(families):
        raise ValueError("Every fold must contain all simple controls")
    totals = {name: {"sse": 0.0, "n_targets": 0} for name in families}
    fold_rmse: dict[str, dict[str, float]] = {}
    for fold, rows in fold_metrics.items():
        if set(rows) != families:
            raise ValueError("All folds must contain exactly the same candidate families")
        counts = set()
        fold_rmse[fold] = {}
        for name, row in rows.items():
            sse, count = float(row["sse"]), row["n_targets"]
            if not np.isfinite(sse) or sse < 0 or not isinstance(count, (int, np.integer)) or count <= 0:
                raise ValueError("Candidate SSE must be finite/nonnegative and count a positive integer")
            counts.add(int(count))
            totals[name]["sse"] += sse
            totals[name]["n_targets"] += int(count)
            fold_rmse[fold][name] = float(np.sqrt(sse / count))
        if len(counts) != 1:
            raise ValueError("Candidate families must score the same target count within each fold")
    aggregate = {
        name: {**totals[name], "rmse": float(np.sqrt(totals[name]["sse"] / totals[name]["n_targets"]))}
        for name in sorted(families)
    }
    if any(not np.isfinite(row["rmse"]) for row in aggregate.values()):
        raise FloatingPointError("Nonfinite pooled candidate errors")
    order = list(simple_families)
    if family_order is not None:
        order.extend(name for name in family_order if name in families and name not in order)
    order.extend(sorted(families - set(order)))

    def best(names: Sequence[str]) -> str:
        chosen = names[0]
        for name in names[1:]:
            if aggregate[name]["rmse"] < aggregate[chosen]["rmse"] - 1e-12:
                chosen = name
        return chosen

    best_simple = best(list(simple_families))
    simple_rmse = aggregate[best_simple]["rmse"]
    gates = {}
    eligible = []
    for name in order:
        if name in simple_families:
            continue
        per_fold = {
            fold: fold_rmse[fold][name] <= 1.10 * min(fold_rmse[fold][simple] for simple in simple_families)
            for fold in fold_metrics
        }
        # If the best control is perfect, a tied learned model offers no gain.
        margin = simple_rmse > 0 and aggregate[name]["rmse"] <= 0.95 * simple_rmse
        passed = bool(margin and all(per_fold.values()))
        gates[name] = {"aggregate_5pct_improvement": bool(margin),
                       "per_fold_within_10pct": per_fold, "passed": passed}
        if passed:
            eligible.append(name)
    selected = best(eligible) if eligible else best_simple
    return {
        "selected_family": selected,
        "best_simple_family": best_simple,
        "learned_selected": bool(eligible),
        "aggregate": aggregate,
        "per_fold_rmse": fold_rmse,
        "per_family_gates": gates,
        "family_order": order,
        "selection_rule": "pooled_sse_5pct_improvement_and_each_fold_within_10pct",
    }
