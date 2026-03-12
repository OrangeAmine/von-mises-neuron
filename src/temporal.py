"""Temporal Von Mises components: time-aware encoding and forecasting.

Extends the Von Mises neuron with explicit temporal features:
  1. Angular time extraction: timestamps → raw angles in [0, 2π)
  2. Frequency bank: frozen-κ multi-scale VM layer
  3. Phase fusion: time-conditioned Δμ shifts for sensor VM neurons
  4. Temporal conv bridges: depthwise Conv1d for local temporal mixing

See docs/superpowers/specs/2026-03-12-temporal-vm-encoder-design.md
"""

import math
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.von_mises_neuron import VonMisesLinear


def extract_angular_time_features(
    index: pd.DatetimeIndex,
    periods: Optional[List[int]] = None,
) -> np.ndarray:
    """Convert a DatetimeIndex into angular time features in [0, 2π).

    Each period maps a cyclic time component to an angle using a unified
    formula: θ = 2π · (fractional_position_within_period).

      - 24   → θ = 2π · hour / 24               (daily cycle)
      - 168  → θ = 2π · (dow·24 + hour) / 168    (weekly cycle, continuous)
      - 8766 → θ = 2π · day_of_year / 365.25     (annual cycle)

    The weekly formula uses continuous hours within the week (not discrete
    dayofweek) so the angle interpolates smoothly between days.

    Args:
        index: DatetimeIndex from a sensor DataFrame.
        periods: List of periods in hours. Default [24, 168, 8766].

    Returns:
        np.ndarray of shape (T, len(periods)), dtype float64, values in [0, 2π).
    """
    if periods is None:
        periods = [24, 168, 8766]

    # Fractional hour within the day (0.0 to ~23.99)
    hours = index.hour + index.minute / 60.0 + index.second / 3600.0

    features = []
    for p in periods:
        if p == 24:
            # Daily: fractional hour → angle
            theta = 2.0 * np.pi * hours / 24.0
        elif p == 168:
            # Weekly: continuous hours within the week → angle
            # dayofweek * 24 + fractional_hour gives position in [0, 168)
            weekly_hours = index.dayofweek * 24.0 + hours
            theta = 2.0 * np.pi * weekly_hours / 168.0
        elif p == 8766:
            # Annual: day-of-year → angle (day_of_year starts at 1)
            theta = 2.0 * np.pi * index.dayofyear / 365.25
        else:
            # Generic: compute hours since start, mod by period
            t_hours = (index - index[0]).total_seconds() / 3600.0
            theta = 2.0 * np.pi * (t_hours % p) / p

        # Wrap to [0, 2π) for safety
        theta = np.asarray(theta, dtype=np.float64) % (2.0 * np.pi)
        features.append(theta)

    return np.column_stack(features)
