"""Tests for circular (angular) utilities.

Covers:
  - Circular distance properties (symmetry, range, identity, known values)
  - Loss functions (cosine distance, sin/cos MSE)
  - Circular metrics (CMAE, CRMSE, CR2)
  - Mixed normalization (angular vs real-valued columns)
  - Sin/cos ↔ angle conversion round-trip
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.circular import (
    circular_distance,
    circular_mae,
    circular_r2,
    circular_rmse,
    compute_circular_metrics,
    cosine_distance_loss,
    mixed_normalize,
    sincos_mse_loss,
    sincos_to_angle,
)


# ---------------------------------------------------------------------------
# Circular distance
# ---------------------------------------------------------------------------


class TestCircularDistance:
    """Tests for the arc distance function."""

    def test_identity(self):
        """Distance between identical angles is zero."""
        a = torch.tensor([0.0, 1.0, math.pi, 5.5])
        d = circular_distance(a, a)
        torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-5, rtol=0)

    def test_symmetry(self):
        """Distance is symmetric: d(a, b) == d(b, a)."""
        a = torch.tensor([0.5, 1.0, 3.0])
        b = torch.tensor([2.0, 5.0, 6.0])
        torch.testing.assert_close(
            circular_distance(a, b),
            circular_distance(b, a),
            atol=1e-6, rtol=0,
        )

    def test_range(self):
        """Distance is always in [0, pi]."""
        a = torch.rand(100) * 2 * math.pi
        b = torch.rand(100) * 2 * math.pi
        d = circular_distance(a, b)
        assert (d >= 0).all()
        assert (d <= math.pi + 1e-6).all()

    def test_opposite_is_pi(self):
        """Opposite angles are pi apart."""
        a = torch.tensor([0.0, 1.0, 2.0])
        b = a + math.pi
        d = circular_distance(a, b)
        torch.testing.assert_close(
            d,
            torch.full_like(d, math.pi),
            atol=1e-5, rtol=0,
        )

    def test_wraparound(self):
        """0.01 rad and 6.27 rad (~2pi - 0.01) are ~0.02 apart, not ~6.26."""
        a = torch.tensor([0.01])
        b = torch.tensor([2 * math.pi - 0.01])
        d = circular_distance(a, b)
        assert d.item() < 0.03, f"Wraparound distance should be ~0.02, got {d.item()}"

    def test_known_values(self):
        """Quarter turn = pi/2, half turn = pi."""
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([math.pi / 2, math.pi])
        d = circular_distance(a, b)
        torch.testing.assert_close(
            d,
            torch.tensor([math.pi / 2, math.pi]),
            atol=1e-5, rtol=0,
        )

    def test_2pi_periodicity(self):
        """Adding 2pi to either argument doesn't change distance."""
        a = torch.rand(50) * 2 * math.pi
        b = torch.rand(50) * 2 * math.pi
        d1 = circular_distance(a, b)
        d2 = circular_distance(a + 2 * math.pi, b)
        d3 = circular_distance(a, b + 4 * math.pi)
        torch.testing.assert_close(d1, d2, atol=1e-5, rtol=0)
        torch.testing.assert_close(d1, d3, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestCosineDistanceLoss:
    """Tests for cosine_distance_loss."""

    def test_zero_for_identical(self):
        """Loss is zero when pred == target."""
        a = torch.tensor([0.5, 1.0, 3.0])
        loss = cosine_distance_loss(a, a)
        assert loss.item() < 1e-6

    def test_max_for_opposite(self):
        """Loss is 2.0 when pred and target are pi apart."""
        a = torch.tensor([0.0, 1.0, 2.0])
        b = a + math.pi
        loss = cosine_distance_loss(a, b)
        assert abs(loss.item() - 2.0) < 1e-5

    def test_range(self):
        """Loss is in [0, 2] for arbitrary inputs."""
        a = torch.rand(100) * 2 * math.pi
        b = torch.rand(100) * 2 * math.pi
        loss = cosine_distance_loss(a, b)
        assert 0 <= loss.item() <= 2.0 + 1e-6

    def test_gradient_exists(self):
        """Loss produces gradients for backprop."""
        pred = torch.tensor([1.0, 2.0], requires_grad=True)
        target = torch.tensor([1.5, 2.5])
        loss = cosine_distance_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.all(pred.grad == 0)


class TestSincosMSELoss:
    """Tests for sincos_mse_loss."""

    def test_zero_for_correct_sincos(self):
        """Loss is zero when sin/cos match the target angle."""
        target = torch.tensor([0.5, 1.0, 3.0])
        sin_pred = torch.sin(target)
        cos_pred = torch.cos(target)
        loss = sincos_mse_loss(sin_pred, cos_pred, target)
        assert loss.item() < 1e-10

    def test_positive_for_wrong_sincos(self):
        """Loss is positive when sin/cos don't match."""
        target = torch.tensor([0.5, 1.0, 3.0])
        sin_pred = torch.zeros_like(target)
        cos_pred = torch.ones_like(target)
        loss = sincos_mse_loss(sin_pred, cos_pred, target)
        assert loss.item() > 0.01

    def test_gradient_flows(self):
        """Gradients flow to both sin_pred and cos_pred."""
        sin_pred = torch.tensor([0.5, 0.3], requires_grad=True)
        cos_pred = torch.tensor([0.8, 0.9], requires_grad=True)
        target = torch.tensor([1.0, 2.0])
        loss = sincos_mse_loss(sin_pred, cos_pred, target)
        loss.backward()
        assert sin_pred.grad is not None
        assert cos_pred.grad is not None


# ---------------------------------------------------------------------------
# Sin/cos conversion
# ---------------------------------------------------------------------------


class TestSincosToAngle:
    """Tests for sincos_to_angle round-trip."""

    def test_round_trip(self):
        """angle -> (sin, cos) -> angle preserves the original."""
        angles = torch.tensor([0.0, 0.5, math.pi, 1.5 * math.pi, 5.0])
        sin_val = torch.sin(angles)
        cos_val = torch.cos(angles)
        recovered = sincos_to_angle(sin_val, cos_val)
        # Compare via circular distance (handles wraparound)
        d = circular_distance(recovered, angles)
        torch.testing.assert_close(d, torch.zeros_like(d), atol=1e-5, rtol=0)

    def test_output_range(self):
        """Output is always in [0, 2pi)."""
        sin_val = torch.rand(100) * 2 - 1
        cos_val = torch.rand(100) * 2 - 1
        angles = sincos_to_angle(sin_val, cos_val)
        assert (angles >= 0).all()
        assert (angles < 2 * math.pi + 1e-6).all()

    def test_known_values(self):
        """Check specific angles."""
        # sin=0, cos=1 → angle=0
        assert abs(sincos_to_angle(torch.tensor(0.0), torch.tensor(1.0)).item()) < 1e-5
        # sin=1, cos=0 → angle=pi/2
        assert abs(
            sincos_to_angle(torch.tensor(1.0), torch.tensor(0.0)).item() - math.pi / 2
        ) < 1e-5
        # sin=0, cos=-1 → angle=pi
        assert abs(
            sincos_to_angle(torch.tensor(0.0), torch.tensor(-1.0)).item() - math.pi
        ) < 1e-5


# ---------------------------------------------------------------------------
# Circular metrics
# ---------------------------------------------------------------------------


class TestCircularMetrics:
    """Tests for CMAE, CRMSE, CR2."""

    def test_perfect_prediction(self):
        """Perfect prediction: CMAE=0, CRMSE=0, CR2=1."""
        target = torch.rand(100) * 2 * math.pi
        metrics = compute_circular_metrics(target, target)
        assert metrics["cmae"] < 1e-5
        assert metrics["crmse"] < 1e-5
        # CR2 should be 1.0 for perfect predictions (with sufficient variance)
        assert metrics["cr2"] > 0.99

    def test_random_prediction_low_r2(self):
        """Random predictions should have low (possibly negative) CR2."""
        torch.manual_seed(42)
        target = torch.rand(500) * 2 * math.pi
        pred = torch.rand(500) * 2 * math.pi
        metrics = compute_circular_metrics(pred, target)
        # Random predictions shouldn't beat the circular mean
        assert metrics["cr2"] < 0.5

    def test_degree_conversion(self):
        """Degree metrics should be 180/pi times radian metrics."""
        target = torch.rand(50) * 2 * math.pi
        pred = target + 0.1 * torch.randn(50)
        metrics = compute_circular_metrics(pred, target)
        assert abs(metrics["cmae_deg"] - math.degrees(metrics["cmae"])) < 1e-5
        assert abs(metrics["crmse_deg"] - math.degrees(metrics["crmse"])) < 1e-5

    def test_metrics_types(self):
        """All metrics are plain Python floats."""
        target = torch.rand(20) * 2 * math.pi
        pred = target + 0.05
        metrics = compute_circular_metrics(pred, target)
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is {type(val)}, expected float"

    def test_cmae_is_nonnegative(self):
        """CMAE is always >= 0."""
        target = torch.rand(100) * 2 * math.pi
        pred = torch.rand(100) * 2 * math.pi
        assert circular_mae(pred, target) >= 0

    def test_crmse_geq_cmae(self):
        """CRMSE >= CMAE (Jensen's inequality for sqrt)."""
        target = torch.rand(100) * 2 * math.pi
        pred = target + 0.3 * torch.randn(100)
        cmae_val = circular_mae(pred, target)
        crmse_val = circular_rmse(pred, target)
        assert crmse_val >= cmae_val - 1e-6


# ---------------------------------------------------------------------------
# Mixed normalization
# ---------------------------------------------------------------------------


class TestMixedNormalize:
    """Tests for mixed_normalize (angular + z-score)."""

    def _make_df(self):
        """Create a test DataFrame with angular and real columns."""
        np.random.seed(42)
        n = 200
        return pd.DataFrame({
            "wind_dir": np.random.uniform(0, 2 * math.pi, n),
            "temperature": np.random.normal(20, 5, n),
            "wind_speed": np.random.normal(10, 3, n),
        })

    def test_angular_cols_in_0_2pi(self):
        """Angular columns are wrapped to [0, 2pi)."""
        df = self._make_df()
        # Inject out-of-range values
        df.loc[0, "wind_dir"] = -0.5
        df.loc[1, "wind_dir"] = 7.0

        norm_df, _ = mixed_normalize(df, angular_cols=["wind_dir"])
        assert (norm_df["wind_dir"] >= 0).all()
        assert (norm_df["wind_dir"] < 2 * math.pi + 1e-6).all()

    def test_real_cols_zscore(self):
        """Real-valued columns are z-scored (approximately mean=0, std=1)."""
        df = self._make_df()
        norm_df, stats = mixed_normalize(df, angular_cols=["wind_dir"])

        for col in ["temperature", "wind_speed"]:
            assert abs(norm_df[col].mean()) < 0.1  # approximately zero
            assert abs(norm_df[col].std() - 1.0) < 0.1  # approximately one
            assert col in stats

    def test_angular_cols_not_in_stats(self):
        """Angular columns don't appear in the stats dict."""
        df = self._make_df()
        _, stats = mixed_normalize(df, angular_cols=["wind_dir"])
        assert "wind_dir" not in stats
        assert "temperature" in stats
        assert "wind_speed" in stats

    def test_stats_reuse(self):
        """Providing stats from training set produces same normalization."""
        df = self._make_df()
        train_df = df.iloc[:150]
        test_df = df.iloc[150:]

        _, stats = mixed_normalize(train_df, angular_cols=["wind_dir"])
        test_norm, _ = mixed_normalize(
            test_df, angular_cols=["wind_dir"], stats=stats,
        )

        # Test set should use train stats, so mean won't be exactly 0
        # but the transformation should still work
        assert len(test_norm) == len(test_df)
        assert "temperature" in stats

    def test_no_angular_cols(self):
        """With no angular columns, behaves like standard z-score."""
        df = self._make_df()
        norm_df, stats = mixed_normalize(df, angular_cols=[])
        # All columns should be in stats
        assert len(stats) == 3

    def test_all_angular_cols(self):
        """With all angular columns, no stats are produced."""
        df = self._make_df()
        norm_df, stats = mixed_normalize(
            df, angular_cols=["wind_dir", "temperature", "wind_speed"],
        )
        assert len(stats) == 0

    def test_original_not_modified(self):
        """Original DataFrame is not modified (returns a copy)."""
        df = self._make_df()
        original_vals = df["temperature"].copy()
        mixed_normalize(df, angular_cols=["wind_dir"])
        pd.testing.assert_series_equal(df["temperature"], original_vals)
