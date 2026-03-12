"""Synthetic data generators for circular/angular domain experiments.

Three domains where angles are physically meaningful and the 2π wrap-around
is a real property of the data, not an artifact:

  1. Wind direction — hourly meteorological data with diurnal/weekly drift
  2. Protein dihedral angles — (phi, psi) on the Ramachandran plot
  3. Robotics joint trajectory — motor angles crossing the 0/2π boundary

All angular values are in radians, range [0, 2π) or [-π, π) as noted.
"""

import math
from typing import Tuple

import numpy as np
import pandas as pd


def generate_wind_data(
    n_hours: int = 4320,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hourly wind data with circular direction.

    The wind direction follows a von Mises distribution with a mean
    that drifts diurnally (24h cycle) and weekly (168h cycle):

        μ(t) = π + 0.5·sin(2π·t/24) + 0.3·sin(2π·t/168)
        θ(t) ~ vonMises(μ(t), κ=3.0)

    κ=3.0 gives moderate directional spread (~33° std dev), which is
    realistic for many mid-latitude locations.

    Two correlated auxiliary features are included:
      - wind_speed: 5 + 2·sin(2π·t/24 + π/6) + N(0, 0.5)
      - temperature: 15 + 8·sin(2π·t/24 − π/4) + N(0, 1.0)

    Args:
        n_hours: Number of hourly samples (default 4320 = 180 days).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [wind_direction, wind_speed, temperature]
        and a DatetimeIndex at hourly frequency.
        wind_direction is in [0, 2π).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours, dtype=np.float64)

    # --- Wind direction: von Mises with drifting mean ---
    # Diurnal drift (24h period) + weekly drift (168h period)
    mu = math.pi + 0.5 * np.sin(2 * math.pi * t / 24) + \
        0.3 * np.sin(2 * math.pi * t / 168)

    # Sample from von Mises (numpy uses mu, kappa parameterization)
    # kappa=3.0 → std dev ≈ 1/sqrt(kappa) ≈ 0.577 rad ≈ 33°
    wind_direction = rng.vonmises(mu, 3.0) % (2 * math.pi)

    # --- Wind speed: correlated with time of day ---
    # Peaks in the afternoon (phase offset π/6 from direction cycle)
    wind_speed = (
        5.0
        + 2.0 * np.sin(2 * math.pi * t / 24 + math.pi / 6)
        + rng.normal(0, 0.5, n_hours)
    )
    wind_speed = np.clip(wind_speed, 0, None)  # wind speed can't be negative

    # --- Temperature: sinusoidal diurnal pattern ---
    # Peaks in early afternoon (phase offset −π/4)
    temperature = (
        15.0
        + 8.0 * np.sin(2 * math.pi * t / 24 - math.pi / 4)
        + rng.normal(0, 1.0, n_hours)
    )

    # Build DataFrame with hourly DatetimeIndex
    index = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame(
        {
            "wind_direction": wind_direction,
            "wind_speed": wind_speed,
            "temperature": temperature,
        },
        index=index,
    )

    print(f"  Wind data: {n_hours} hours ({n_hours / 24:.0f} days)")
    print(f"    wind_direction: [{wind_direction.min():.3f}, "
          f"{wind_direction.max():.3f}] rad")
    print(f"    wind_speed: [{wind_speed.min():.2f}, "
          f"{wind_speed.max():.2f}] m/s")
    print(f"    temperature: [{temperature.min():.2f}, "
          f"{temperature.max():.2f}] °C")

    return df


def generate_ramachandran_data(
    n_samples: int = 5000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic protein dihedral angles from a Ramachandran plot.

    The Ramachandran plot describes the allowed (phi, psi) backbone dihedral
    angles in proteins. We model 4 structural clusters:

        Cluster       phi(°)  psi(°)  Weight  kappa
        ─────────────────────────────────────────────
        Alpha-helix    −57     −47     0.35    8.0
        Beta-sheet    −120     120     0.30    6.0
        Left-helix     +60     +40     0.10    5.0
        Random coil  uniform uniform  0.25     —

    Each cluster samples (phi, psi) independently from von Mises
    distributions centered at the cluster mean (or uniform for random coil).

    This is a non-sequential regression task: given phi, predict psi.
    The VM neuron should excel because both input and output live on the
    circle, and the phi→psi mapping has multiple circular modes.

    Args:
        n_samples: Total number of (phi, psi) pairs.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (phi, psi) arrays, each shape (n_samples,),
        both in [-π, π) radians.
    """
    rng = np.random.default_rng(seed)

    # Cluster definitions: (phi_deg, psi_deg, weight, kappa)
    clusters = [
        (-57,  -47,  0.35, 8.0),   # Alpha-helix
        (-120,  120, 0.30, 6.0),    # Beta-sheet
        (60,    40,  0.10, 5.0),    # Left-handed helix (rare)
        (None,  None, 0.25, None),  # Random coil (uniform)
    ]

    weights = [c[2] for c in clusters]
    # Assign each sample to a cluster
    cluster_ids = rng.choice(len(clusters), size=n_samples, p=weights)

    phi = np.zeros(n_samples)
    psi = np.zeros(n_samples)

    for i, (phi_deg, psi_deg, _, kappa) in enumerate(clusters):
        mask = cluster_ids == i
        n_cluster = mask.sum()

        if kappa is not None:
            # Structured cluster: von Mises around the mean angle
            phi_rad = math.radians(phi_deg)
            psi_rad = math.radians(psi_deg)
            phi[mask] = rng.vonmises(phi_rad, kappa, n_cluster)
            psi[mask] = rng.vonmises(psi_rad, kappa, n_cluster)
        else:
            # Random coil: uniform on [-π, π)
            phi[mask] = rng.uniform(-math.pi, math.pi, n_cluster)
            psi[mask] = rng.uniform(-math.pi, math.pi, n_cluster)

    # Wrap to [-π, π) for consistency with biochemistry convention
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    psi = (psi + math.pi) % (2 * math.pi) - math.pi

    # Shuffle to remove cluster ordering
    perm = rng.permutation(n_samples)
    phi = phi[perm]
    psi = psi[perm]

    print(f"  Ramachandran data: {n_samples} (phi, psi) pairs")
    print(f"    Cluster distribution: "
          f"alpha={(cluster_ids == 0).sum()}, "
          f"beta={(cluster_ids == 1).sum()}, "
          f"left-helix={(cluster_ids == 2).sum()}, "
          f"coil={(cluster_ids == 3).sum()}")
    print(f"    phi range: [{math.degrees(phi.min()):.1f}, "
          f"{math.degrees(phi.max()):.1f}]°")
    print(f"    psi range: [{math.degrees(psi.min()):.1f}, "
          f"{math.degrees(psi.max()):.1f}]°")

    return phi, psi


def generate_robot_trajectory(
    n_steps: int = 5000,
    period: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic robot arm joint trajectory with full rotations.

    A robot arm performing continuous rotations — the trajectory regularly
    crosses the 0/2π boundary. Standard models see this as a discontinuous
    jump (6.28 → 0.01), while the VM neuron handles it natively.

    Joint angles:
        joint1(t) = (2π·t/period + offset) % 2π + vonMises_noise(κ=20)
        joint2(t) = joint1(t + period/4) + vonMises_noise(κ=25)

    joint2 is coupled to joint1 with a quarter-period phase offset,
    modeling a coordinated multi-joint motion.

    κ=20 → very tight noise (~0.22 rad ≈ 13° std dev), realistic for
    a servo motor with encoder noise.

    Args:
        n_steps: Number of timesteps.
        period: Steps for one full rotation (default 100).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns [joint1_angle, joint2_angle], both in [0, 2π),
        indexed by integer timestep.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)

    offset = 0.7  # arbitrary phase offset to avoid starting at 0

    # --- Joint 1: smooth rotation + tight von Mises noise ---
    # Base trajectory: linear sweep through [0, 2π) with period
    joint1_base = (2 * math.pi * t / period + offset) % (2 * math.pi)
    # Add noise sampled from von Mises(0, κ=20) — centered, tight
    joint1_noise = rng.vonmises(0, 20.0, n_steps)
    joint1 = (joint1_base + joint1_noise) % (2 * math.pi)

    # --- Joint 2: coupled to joint1 with quarter-period lag ---
    # This lag means when joint1 is at 3 o'clock, joint2 is at 12 o'clock
    joint2_base = (2 * math.pi * (t + period / 4) / period + offset) % (2 * math.pi)
    joint2_noise = rng.vonmises(0, 25.0, n_steps)
    joint2 = (joint2_base + joint2_noise) % (2 * math.pi)

    # Count 0/2π boundary crossings to verify the trajectory wraps
    # (a crossing = the base angle jumps from >5.5 to <1 or vice versa)
    crossings_j1 = np.sum(np.abs(np.diff(joint1_base)) > math.pi)

    df = pd.DataFrame(
        {
            "joint1_angle": joint1,
            "joint2_angle": joint2,
        },
        index=pd.RangeIndex(n_steps, name="timestep"),
    )

    print(f"  Robot trajectory: {n_steps} steps, period={period}")
    print(f"    joint1 0/2pi crossings: {crossings_j1}")
    print(f"    joint1 range: [{joint1.min():.4f}, {joint1.max():.4f}] rad")
    print(f"    joint2 range: [{joint2.min():.4f}, {joint2.max():.4f}] rad")

    return df
