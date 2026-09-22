"""Numerical parity and synchronized reference/factored engineering profiles."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.circular_layers_v2 import PolarCircularLinearV2
from src.methodology_protocol import CONFIG_PATH, environment, file_hash, read_json, write_json


def synchronize(device: torch.device) -> None:
    """Finish pending device work before reading a host clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parity_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Check outputs and all retained-coordinate gradients on finite inputs."""
    rows = []
    for name, dtype in [("cpu", torch.float64), ("cpu", torch.float32),
                        *([("cuda", torch.float32)] if torch.cuda.is_available() else [])]:
        tolerance = config["engineering"]["parity"][str(dtype).split(".")[1]]
        for response in ("sum", "exp"):
            for kappa in (0.001, 0.4):
                for bias in (False, True):
                    for dynamic in (False, True):
                        torch.manual_seed(17)
                        reference = PolarCircularLinearV2(3, 4, bias=bias, response=response,
                            init_kappa=kappa, implementation="reference", dtype=dtype, device=name)
                        factored = copy.deepcopy(reference)
                        factored.implementation = "factored"
                        x = torch.randn(2, 5, 3, dtype=dtype, device=name, requires_grad=True)
                        delta = torch.randn(2, 5, 4, 1, dtype=dtype, device=name, requires_grad=True) if dynamic else None
                        output = reference(x, delta)
                        other = factored(x, delta)
                        cotangent = torch.randn_like(output)
                        shared = [x] + ([delta] if dynamic else [])
                        first = torch.autograd.grad((output*cotangent).sum(), shared + list(reference.parameters()))
                        second = torch.autograd.grad((other*cotangent).sum(), shared + list(factored.parameters()))
                        pairs = [(output, other), *zip(first, second)]
                        rows.append({"device": name, "dtype": str(dtype), "response": response,
                            "init_kappa": kappa, "bias": bias, "dynamic_phase": dynamic,
                            "tolerances": tolerance,
                            "maximum_absolute_output_difference": float((output-other).abs().max().detach()),
                            "maximum_absolute_gradient_difference": max(float((a-b).abs().max()) for a,b in zip(first,second)),
                            "passed": all(bool(torch.isfinite(a).all() and torch.isfinite(b).all()) and torch.allclose(a,b,**tolerance) for a,b in pairs)})
    return rows


def profile(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Measure all prescribed shapes with fixed CPU threads and CUDA sync."""
    rows = []
    spec = config["engineering"]["profile"]
    for name in ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]:
        device = torch.device(name)
        for batch, steps, inputs, outputs in spec["shapes"]:
            for implementation in ("reference", "factored"):
                row = {"device": name, "dtype": "float32", "shape": [batch,steps,inputs,outputs],
                       "implementation": implementation, "response": "sum", "dynamic_phase": True,
                       "warmup": spec["warmup"], "iterations": spec["iterations"], "status": "running"}
                rows.append(row)
                try:
                    torch.manual_seed(17)
                    if name == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    model = PolarCircularLinearV2(inputs,outputs,implementation=implementation,device=device)
                    x = torch.randn(batch,steps,inputs,device=device,requires_grad=True)
                    delta = torch.randn(batch,steps,outputs,1,device=device,requires_grad=True)
                    def iteration() -> None:
                        model.zero_grad(set_to_none=True)
                        x.grad = delta.grad = None
                        model(x,delta).square().mean().backward()
                    for _ in range(spec["warmup"]):
                        iteration()
                    synchronize(device)
                    if name == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    timings = []
                    for _ in range(spec["iterations"]):
                        synchronize(device)
                        start = time.perf_counter()
                        iteration()
                        synchronize(device)
                        timings.append(1000*(time.perf_counter()-start))
                    row.update(status="completed", median_ms=float(np.median(timings)),
                               p90_ms=float(np.percentile(timings,90)),
                               peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if name == "cuda" else None)
                    del model, x, delta
                except torch.OutOfMemoryError as exc:
                    row.update(status="out_of_memory", error=str(exc))
                print(row, flush=True)
    return rows


def main() -> int:
    """Write a new benchmark report; never overwrite a prior measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(CONFIG_PATH)
    report = {"environment": environment(), "source_hashes": {
        name: file_hash(ROOT/name) for name in ("src/circular_layers_v2.py", "scripts/benchmark_circular_layers_v2.py")}}
    report["parity"] = parity_cases(config)
    report["profiles"] = profile(config)
    # Exponential overflow is a retained limitation, not a clamped parity pass.
    extreme = PolarCircularLinearV2(1,1,response="exp",init_kappa=1000)
    with torch.no_grad():
        extreme.phase.zero_()
        value = extreme(torch.zeros(1,1))
    report["extreme_exp_diagnostic"] = {"requested_kappa": 1000,
        "output_finite": bool(torch.isfinite(value).all()), "classification": "expected_unbounded_exp_failure_regime"}
    report["all_parity_passed"] = all(row["passed"] for row in report["parity"])
    write_json(args.output, report)
    return 0 if report["all_parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
