"""Small fixed-calendar source estimators for N06; diagnostic eligibility only."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.dynamics_models_v1 import CalendarBasisV1


class CalendarSourceV1:
    """Fit training-only daily-bin means or a fixed calendar ridge readout.

    Args:
        mode: seasonal, vm or fourier; no adaptive kernel geometry.
        spec: Feature count, fixed concentration and ridge coefficient.
        device: Optional explicit device; defaults to CUDA when available.

    This estimator takes clock coordinates, not controlled sensor histories.
    Source predictions remain action-independent, conditional on stable routines.
    It is not the N04 neural adapter and does not inherit its acceptance.
    """

    def __init__(self, mode: str, spec: dict[str, Any], device: str | None = None):
        if mode not in ('seasonal', 'vm', 'fourier'):
            raise ValueError('Unknown source estimator')
        if spec['ridge'] <= 0:
            raise ValueError('Positive ridge required')
        self.mode, self.spec = mode, dict(spec)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.basis = None if mode == 'seasonal' else CalendarBasisV1(
            'fixed' if mode == 'vm' else 'fourier', spec['features'], spec['kappa'],
            device=self.device, dtype=torch.float64)
        self.coefficients: torch.Tensor | None = None
        self.hour_means: np.ndarray | None = None

    def _design(self, hours: np.ndarray) -> torch.Tensor:
        """Map raw UTC hours to an intercept plus periodic features."""
        theta = torch.as_tensor(np.asarray(hours) % 24 * (2 * math.pi / 24),
                                device=self.device, dtype=torch.float64)
        features = self.basis(theta)
        return torch.cat((torch.ones_like(theta[..., None]), features), dim=-1)

    def fit(self, hours: np.ndarray, labels: np.ndarray) -> 'CalendarSourceV1':
        """Fit from explicitly supplied training rows only; no model selection."""
        x, y = np.asarray(hours), np.asarray(labels)
        if x.ndim != 1 or y.shape != x.shape or not len(x) or not np.isfinite([x, y]).all():
            raise ValueError('Training hours/labels must be finite paired vectors')
        if self.mode == 'seasonal':
            slots = np.rint((x % 24) * 4).astype(int) % 96
            count = np.bincount(slots, minlength=96)
            total = np.bincount(slots, weights=y, minlength=96)
            self.hour_means = np.divide(total, count, out=np.full(96, y.mean()), where=count > 0)
        else:
            with torch.no_grad():
                design = self._design(x)
                target = torch.as_tensor(y, device=self.device, dtype=torch.float64)
                penalty = torch.eye(design.shape[1], device=self.device, dtype=torch.float64) * self.spec['ridge']
                penalty[0, 0] = 0
                self.coefficients = torch.linalg.solve(design.T @ design / len(y) + penalty,
                                                       design.T @ target / len(y))
        return self

    def predict(self, hours: np.ndarray) -> np.ndarray:
        """Return nonnegative point forecasts with the input clock-array shape."""
        x = np.asarray(hours)
        if not np.isfinite(x).all():
            raise ValueError('Prediction hours must be finite')
        if self.mode == 'seasonal':
            if self.hour_means is None:
                raise ValueError('Fit before predict')
            result = self.hour_means[np.rint((x % 24) * 4).astype(int) % 96]
        else:
            if self.coefficients is None:
                raise ValueError('Fit before predict')
            with torch.no_grad():
                result = (self._design(x) @ self.coefficients).cpu().numpy()
        return np.maximum(0, result)

    def record(self) -> dict[str, Any]:
        """Return all fitted coefficients and eligibility for reproducibility."""
        values = self.hour_means if self.mode == 'seasonal' else self.coefficients.cpu().numpy()
        return {'mode': self.mode, 'fitted_coefficients': len(values), 'values': values.tolist(),
                'fixed_geometry': self.spec, 'device': str(self.device),
                'eligibility': 'synthetic diagnostic; see separate standard acceptance'}
