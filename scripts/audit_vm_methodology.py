"""Check two VM algebra identities in float64 without fitting or editing models.

This diagnostic checks numerical output/gradient agreement, not forecasting
quality, optimization equivalence, training stability, or execution speed.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.temporal import TemporalVonMisesEncoder
from src.von_mises_neuron import VonMisesLinear

ATOL = 1e-12
RTOL = 1e-12


def comparison(actual: Tensor, expected: Tensor) -> dict[str, Any]:
    """Summarize finite tensor agreement under the declared tolerances."""
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    return {"shape": list(actual.shape), "finite": finite,
            "max_absolute_error": float((actual - expected).abs().max()) if finite else None,
            "passed": finite and torch.allclose(actual, expected, atol=ATOL, rtol=RTOL)}


def audit(seed: int = 7) -> dict[str, Any]:
    """Compare dynamic VM factorization and temporal-bank projection collapse.

    Args:
        seed: CPU initialization/input seed; no optimizer is constructed.

    Returns:
        JSON-compatible per-identity output/gradient errors and provenance.
    """
    torch.manual_seed(seed)
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "dtype": "torch.float64", "device": "cpu", "torch": torch.__version__,
        "python": sys.version, "atol": ATOL, "rtol": RTOL,
        "scope": "Two algebra identities only; no training, performance benchmark, forecasting validation, or claim of equal optimizer trajectories",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [Path(__file__), ROOT / "src/von_mises_neuron.py", ROOT / "src/temporal.py"]},
        "dynamic_phase_factorization": [], "temporal_projection_collapse": [],
    }
    for log_output in (True, False):
        layer = VonMisesLinear(3, 4, log_output=log_output).double()
        x = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        delta = torch.randn(2, 5, 4, 1, dtype=torch.float64, requires_grad=True)
        kappa = F.softplus(layer.log_kappa)
        a, b = kappa * layer.mu.cos(), -kappa * layer.mu.sin()
        real = F.linear(x.cos(), a) + F.linear(x.sin(), b)
        imaginary = F.linear(x.sin(), a) - F.linear(x.cos(), b)
        # cos(u + delta) = cos(u)cos(delta) - sin(u)sin(delta).
        z = real * delta.squeeze(-1).cos() - imaginary * delta.squeeze(-1).sin()
        factored = (z if log_output else z.exp()) + layer.bias
        original = layer(x, phase_shift=delta)
        # A common random cotangent checks nonuniform output contributions.
        cotangent = torch.randn_like(original)
        variables = {"x": x, "mu": layer.mu, "raw_kappa": layer.log_kappa,
                     "delta": delta, "bias": layer.bias}
        original_grads = torch.autograd.grad((original * cotangent).sum(), tuple(variables.values()),
                                             retain_graph=True)
        factored_grads = torch.autograd.grad((factored * cotangent).sum(), tuple(variables.values()))
        report["dynamic_phase_factorization"].append({
            "log_output": log_output, "output": comparison(factored, original),
            "gradients": {name: comparison(actual, expected)
                          for name, actual, expected in zip(variables, factored_grads, original_grads)},
        })
    encoder = TemporalVonMisesEncoder(sensor_size=3, time_size=2, hidden_size=8,
                                      num_vm_layers=2).double()
    bank = encoder.freq_bank
    times = torch.randn(2, 5, 2, dtype=torch.float64)
    a = bank.get_kappa()[:, None] * bank.mu.cos()
    b = -bank.get_kappa()[:, None] * bank.mu.sin()
    for index, projection in enumerate(encoder.delta_mu_projections):
        original = projection(bank(times))
        # Compose the bank's trigonometric linear map with each affine projection.
        collapsed = (F.linear(times.cos(), projection.weight @ a)
                     + F.linear(times.sin(), projection.weight @ b)
                     + projection.weight @ bank.bias + projection.bias)
        report["temporal_projection_collapse"].append({
            "projection_index": index, "output": comparison(collapsed, original)})
    checks = [case["output"]["passed"] for case in report["dynamic_phase_factorization"]]
    checks.extend(check["passed"] for case in report["dynamic_phase_factorization"]
                  for check in case["gradients"].values())
    checks.extend(case["output"]["passed"] for case in report["temporal_projection_collapse"])
    report["all_passed"] = all(checks)
    return report


def main() -> int:
    """Write a fresh diagnostic JSON and return failure for any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = audit(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"all_passed": report["all_passed"], "output": str(args.output)}, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
