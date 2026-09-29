"""Export a deterministic source-envelope/action-path engineering fixture."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.decision_plant_v1 import Plant, exact_update
from src.source_interval_bridge_v1 import (
    FlowPlan, ReceivedState, calibrate_source_envelope, classify_limit, project_source_envelope,
)


def example() -> dict:
    """Verify conditional containment for two plans with a shared locked prefix."""
    plant = Plant()
    step = round(plant.dt_hours * 3_600_000_000_000)
    issue, horizon = 1000 * step, 9
    point = np.full(horizon, 60000.0)
    ci = (np.arange(20, dtype=np.int64) * 12 + 100) * step
    ce = ci[:, None] + np.arange(horizon, dtype=np.int64) * step
    cp = np.tile(point, (len(ci), 1))
    cy = cp + np.linspace(-3000, 3000, len(ci))[:, None]
    source = calibrate_source_envelope(point=point, issue_ns=issue, start_ns=issue-step, step_ns=step,
        model_id="illustrative-fixed-source-v1", calibration_prediction=cp, calibration_truth=cy,
        calibration_issue_ns=ci, calibration_end_ns=ce, calibration_arrival_ns=ce+step)
    state = ReceivedState(issue-step, issue, 590.0, 610.0)
    prefix = np.full(1 + plant.actuator_delay_steps, plant.base_flow_m3h)
    plans = []
    for label, future_flow in (("base_after_queue", plant.base_flow_m3h),
                               ("high_after_queue", plant.base_flow_m3h + plant.extra_flow_m3h)):
        flow = np.concatenate((prefix, np.full(horizon-len(prefix), future_flow)))
        plan = FlowPlan(label, flow, prefix)
        result = project_source_envelope(source, state, plan, plant)
        # Fixture truth is deliberately bounded: this checks propagation only.
        truth_source = point + np.linspace(-2000, 2000, horizon)
        truth, c = [], 600.0
        for s, q in zip(truth_source, flow):
            c = exact_update(c, s, q, plant)
            truth.append(c)
        truth = np.array(truth)[1:]
        contained = bool(np.all(truth >= result.lower_ppm) and np.all(truth <= result.upper_ppm))
        if not contained:
            raise AssertionError("conditional containment fixture failed")
        plans.append({"plan": asdict(plan), "envelope": asdict(result), "truth_ppm_fixture_only": truth,
                      "contained": contained, "limit_signal": classify_limit(result, plant.limit_ppm)})
    return {"schema_version": "source-interval-bridge-v1", "material_passport": "ENGINEERING EXAMPLE",
            "scope": "deterministic conditional containment, not a coverage or policy study",
            "plant": asdict(plant), "source": asdict(source), "received_state": asdict(state), "plans": plans}


def main() -> None:
    """Write a new fixture receipt without overwriting an earlier artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = example()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False,
                  default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())
        stream.write("\n")
    print(json.dumps({"contained": all(p["contained"] for p in result["plans"]),
                      "plans": {p["plan"]["plan_id"]: p["limit_signal"] for p in result["plans"]}}, indent=2))


if __name__ == "__main__":
    main()
