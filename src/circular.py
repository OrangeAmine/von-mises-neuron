"""Circular (angular) utilities: metrics, losses, and mixed normalization.

Standard metrics treat angles naively — e.g. MSE sees 1° and 359° as 358°
apart, when they're actually 1° apart on the circle. Every function here
respects the topology of S¹ (the unit circle).

Key concepts:
  - Circular distance: shortest arc via arccos(cos(a - b)), range [0, π]
  - Circular mean: atan2(mean(sin(θ)), mean(cos(θ)))
  - Sin/cos representation: angles encoded as (sin(θ), cos(θ)) to avoid
    the 0/2π discontinuity in model outputs
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Circular distance
# ---------------------------------------------------------------------------


def circular_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Shortest arc distance between two angles, element-wise.

    Uses arccos(cos(a - b)) which gives the unsigned shortest arc in [0, π].
    This is more numerically stable than manual wrapping with modular
    arithmetic, and handles all quadrants correctly.

    Args:
        pred: Predicted angles (radians), any shape.
        target: Target angles (radians), same shape as pred.

    Returns:
        Element-wise arc distance in [0, π], same shape as inputs.
    """
    # cos(a - b) is in [-1, 1]; clamp for numerical safety in acos.
    # We clamp to exactly [-1, 1] (not [-1+eps, 1-eps]) to avoid
    # introducing measurable error — acos(1 - 1e-7) ≈ 4.5e-4, not 0.
    return torch.acos(torch.clamp(torch.cos(pred - target), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Loss functions (for training)
# ---------------------------------------------------------------------------


def cosine_distance_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Cosine distance loss: mean(1 - cos(pred - target)).

    Range [0, 2]. Smooth everywhere — no gradient discontinuities.
    Gradient w.r.t. pred = sin(pred - target), which is continuous and
    zero at the correct angle. Preferred over wrapping-based MSE for
    angular regression.

    Args:
        pred: Predicted angles (radians).
        target: Target angles (radians).

    Returns:
        Scalar loss tensor.
    """
    return (1.0 - torch.cos(pred - target)).mean()


def sincos_mse_loss(
    sin_pred: torch.Tensor,
    cos_pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE between predicted (sin, cos) pair and target angle.

    Computes: mean( (sin_pred - sin(target))^2 + (cos_pred - cos(target))^2 )

    Used when the model outputs sin/cos components directly (the preferred
    output strategy for angular regression, since it avoids the 0/2π
    discontinuity in the output space).

    Args:
        sin_pred: Predicted sin component.
        cos_pred: Predicted cos component.
        target: Target angles (radians).

    Returns:
        Scalar loss tensor.
    """
    sin_target = torch.sin(target)
    cos_target = torch.cos(target)
    return (
        (sin_pred - sin_target).pow(2) + (cos_pred - cos_target).pow(2)
    ).mean()


# ---------------------------------------------------------------------------
# Sin/cos helpers
# ---------------------------------------------------------------------------


def sincos_to_angle(
    sin_val: torch.Tensor,
    cos_val: torch.Tensor,
) -> torch.Tensor:
    """Convert (sin, cos) pair back to angle in [0, 2π).

    Uses atan2 for correct quadrant resolution, then shifts from
    [-π, π) to [0, 2π) via modular arithmetic.

    Args:
        sin_val: Sin component, any shape.
        cos_val: Cos component, same shape.

    Returns:
        Angles in [0, 2π), same shape.
    """
    return torch.atan2(sin_val, cos_val) % (2.0 * math.pi)


# ---------------------------------------------------------------------------
# Circular metrics (for evaluation)
# ---------------------------------------------------------------------------


def circular_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Circular Mean Absolute Error — mean shortest arc distance (radians).

    Args:
        pred: Predicted angles (radians).
        target: Target angles (radians).

    Returns:
        CMAE in radians (scalar float).
    """
    return circular_distance(pred, target).mean().item()


def circular_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Circular RMSE — sqrt(mean(arc_distance²)) (radians).

    Args:
        pred: Predicted angles (radians).
        target: Target angles (radians).

    Returns:
        CRMSE in radians (scalar float).
    """
    return circular_distance(pred, target).pow(2).mean().sqrt().item()


def circular_r2(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Circular R² — coefficient of determination for angular data.

    R² = 1 - SS_res / SS_tot, where:
      - SS_res = Σ arc_dist(pred_i, target_i)²
      - SS_tot = Σ arc_dist(target_i, circular_mean(target))²

    The circular mean is computed via: atan2(mean(sin(θ)), mean(cos(θ))),
    which is the proper analog of the arithmetic mean on the circle.

    Returns:
        CR² (scalar float). Can be negative if model is worse than
        predicting the circular mean.
    """
    arc_res = circular_distance(pred, target)
    ss_res = arc_res.pow(2).sum()

    # Circular mean of targets: the "constant model" baseline
    # atan2(mean(sin), mean(cos)) gives the direction of the mean resultant
    circ_mean = torch.atan2(
        torch.sin(target).mean(),
        torch.cos(target).mean(),
    )
    arc_tot = circular_distance(target, circ_mean.expand_as(target))
    ss_tot = arc_tot.pow(2).sum()

    # Avoid division by zero (all targets identical → perfect baseline → R²=0)
    if ss_tot < 1e-10:
        return 0.0

    return (1.0 - ss_res / ss_tot).item()


def compute_circular_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Compute all circular metrics in one call.

    Args:
        pred: Predicted angles (radians).
        target: Target angles (radians).

    Returns:
        Dict with keys: cmae, crmse, cr2, cmae_deg, crmse_deg.
    """
    cmae = circular_mae(pred, target)
    crmse = circular_rmse(pred, target)
    cr2 = circular_r2(pred, target)

    return {
        "cmae": cmae,
        "crmse": crmse,
        "cr2": cr2,
        "cmae_deg": math.degrees(cmae),
        "crmse_deg": math.degrees(crmse),
    }


# ---------------------------------------------------------------------------
# Mixed normalization (angular + real-valued columns)
# ---------------------------------------------------------------------------


def mixed_normalize(
    df: pd.DataFrame,
    angular_cols: List[str],
    stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Normalize a DataFrame with mixed angular and real-valued columns.

    Angular columns: wrapped to [0, 2π) — NO z-score, because centering
    and scaling destroys the circular topology (1 rad and 6.27 rad are
    neighbors, but z-score treats them as ~5.3 standard deviations apart).

    Real-valued columns: standard z-score normalization, same as
    ``preprocessing.normalize()``.

    When ``stats`` is None (training set), computes z-score stats from the
    real-valued columns. When ``stats`` is provided (val/test), applies
    those stats without recomputing.

    Args:
        df: DataFrame with sensor columns.
        angular_cols: List of column names that contain angular data
            (values in radians). These will be wrapped to [0, 2π) and
            NOT z-scored.
        stats: Optional dict of {col: (mean, std)} for real-valued columns.
            Ignored for angular columns. If None, computed from df.

    Returns:
        Tuple of (normalized_df, stats_dict). Stats dict contains entries
        only for real-valued columns (angular columns don't need stats).

    Example:
        train_norm, stats = mixed_normalize(train_df, angular_cols=["wind_dir"])
        val_norm, _ = mixed_normalize(val_df, angular_cols=["wind_dir"], stats=stats)
    """
    df = df.copy()
    computed_stats: Dict[str, Tuple[float, float]] = {}

    for col in df.columns:
        if col in angular_cols:
            # Wrap to [0, 2π) — preserves circular topology
            df[col] = df[col] % (2.0 * math.pi)
        else:
            # Standard z-score for real-valued columns
            if stats is None:
                mean = df[col].mean()
                std = df[col].std()
                if std == 0 or np.isnan(std):
                    std = 1.0
            else:
                mean, std = stats[col]

            df[col] = (df[col] - mean) / std
            computed_stats[col] = (float(mean), float(std))

    # Logging
    if angular_cols:
        print(f"  Angular columns (wrapped to [0, 2pi)): {angular_cols}")
    if computed_stats:
        if stats is None:
            print("  Z-score stats computed from data (training set):")
        else:
            print("  Z-score stats applied from training set:")
        for col, (m, s) in computed_stats.items():
            print(f"    {col}: mean={m:.4f}, std={s:.4f}")

    return df, computed_stats
