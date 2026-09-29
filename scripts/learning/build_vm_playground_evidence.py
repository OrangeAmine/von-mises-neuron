"""Export audited, compact historical evidence for the offline VM playground.

This script reads only tracked documentation. It never imports research models,
fits a model, changes a scientific decision, or reads ignored experimental arrays.
Run from any directory; ``--check`` verifies the committed export and source hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path("learning/vm_playground/evidence.js")
SOURCE_PATHS = {
    "record": "docs/vm_locality_v2_run_record.json",
    "cells": "docs/vm_locality_v2_cells.csv",
    "report_v2": "docs/vm_locality_v2_results.md",
    "report_v1": "docs/vm_methodology_results.md",
    "protocol": "docs/vm_locality_v2_protocol.md",
}
# Verified against the immutable 20260922T112135Z launch manifest. Updating a
# teaching export must never silently adopt edited historical research evidence.
SOURCE_SHA256 = {
    "record": "42d48e103f92c17eef46585f5567d8281ec1171048629352f5c71fa777abd344",
    "cells": "3c2bb09141dff62cd2522beda60ae2937246f88eec34f5dc4e6f612b272e153c",
    "report_v2": "aa8b1ce414e6dceec12603177eff77e47eb8dac8f3521dcb788f83cd1a246dde",
    "report_v1": "836844ed7b6a7202217e5ddeed010502e53ecaf22c5b6114f7561bd3e8755b55",
    "protocol": "59a78e50335a8fb4359ac248ad7681655c6c91d301e3e9c556b18186cad3e52e",
}


def reference(pointer: str, source_id: str = "record") -> dict[str, str]:
    """Return an exact JSON Pointer or a named report section reference."""
    return {"id": source_id, "pointer": pointer}


def _cell_key(row: dict[str, Any], allowance: float | None = None) -> tuple:
    """Identify one complete model/family/noise/label/time cell."""
    return (
        float(row["allowance_seconds"] if allowance is None else allowance),
        row["model_id"],
        row["family"],
        float(row["noise_std"]),
        int(row["total_labels"]),
    )


def _close(actual: float, expected: float, description: str) -> None:
    """Reject nonfinite values and discrepancies beyond summary roundoff."""
    if not (math.isfinite(actual) and math.isfinite(expected)):
        raise ValueError(f"Nonfinite evidence: {description}")
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError(f"Evidence mismatch: {description}: {actual} != {expected}")


def _contrast(value: dict[str, Any], pointer: str) -> dict[str, Any]:
    """Copy all saved contrast statistics, omitting per-target difference arrays."""
    return {
        **{key: item for key, item in value.items() if key != "paired_target_differences"},
        "source": reference(pointer),
    }


def build_evidence(root: Path = ROOT) -> dict[str, Any]:
    """Build a deterministic export and cross-check both compact source formats.

    Args:
        root: Repository root containing the tracked historical documentation.

    Returns:
        Plain JSON data with full cell coverage, saved summaries, and provenance.

    Raises:
        ValueError: If evidence has missing, duplicated, inconsistent, or invalid cells.
    """
    raw = {key: (root / path).read_bytes() for key, path in SOURCE_PATHS.items()}
    for key, content in raw.items():
        if hashlib.sha256(content).hexdigest() != SOURCE_SHA256[key]:
            raise ValueError(f"Historical source differs from inherited snapshot: {SOURCE_PATHS[key]}")
    record = json.loads(raw["record"])
    config = record["config"]
    confirmation = record["phases"]["confirmation"]
    tasks = config["tasks"]
    sources = {
        key: {
            "path": path,
            "href": "../../" + path,
            "sha256": hashlib.sha256(raw[key]).hexdigest(),
        }
        for key, path in SOURCE_PATHS.items()
    }
    axes = {
        "families": tasks["families"],
        "periodic_families": tasks["periodic_families"],
        "noise_levels": tasks["noise_levels"],
        "total_labels": tasks["label_budgets"],
        "allowance_seconds": config["training"]["time_budgets_seconds"],
    }
    models = [
        {**model, "source": reference(f"/config/models/{index}")}
        for index, model in enumerate(config["models"])
    ]
    expected_keys = set(itertools.product(
        axes["allowance_seconds"], [model["id"] for model in models],
        axes["families"], axes["noise_levels"], axes["total_labels"],
    ))
    json_cells: dict[tuple, tuple[dict, str]] = {}
    summaries = []
    for allowance_key, allowance_data in confirmation["time_allowances"].items():
        prefix = f"/phases/confirmation/time_allowances/{allowance_key}"
        if not (allowance_data["complete"] and allowance_data["matrix_complete"]):
            raise ValueError(f"Incomplete historical allowance: {allowance_key}")
        for index, cell in enumerate(allowance_data["cells"]):
            key = _cell_key(cell, float(allowance_key))
            if key in json_cells:
                raise ValueError(f"Duplicate JSON evidence cell: {key}")
            json_cells[key] = cell, f"{prefix}/cells/{index}"
        for model in models:
            model_id = model["id"]
            saved = allowance_data["models"][model_id]
            summaries.append({
                "allowance_seconds": float(allowance_key),
                "model_id": model_id,
                "balanced_periodic_error": saved["balanced_periodic_error"],
                "family_errors": saved["family_errors"],
                "task_sd_by_family": saved["task_sd_by_family"],
                "source": reference(f"{prefix}/models/{model_id}"),
            })
    if set(json_cells) != expected_keys:
        raise ValueError("JSON evidence does not cover the full declared cell matrix")

    cells = []
    seen = set()
    required = tasks["replicates"]["confirmation"] * tasks["initializations"]["confirmation"]
    reader = csv.DictReader(io.StringIO(raw["cells"].decode("utf-8-sig")))
    for csv_row in reader:
        key = _cell_key(csv_row)
        if key in seen:
            raise ValueError(f"Duplicate CSV evidence cell: {key}")
        if key not in json_cells:
            raise ValueError(f"Unexpected CSV evidence cell: {key}")
        seen.add(key)
        saved, pointer = json_cells[key]
        error = float(csv_row["normalized_clean_mse"])
        _close(error, saved["mean_error"], f"CSV line {reader.line_num}")
        if error < 0:
            raise ValueError(f"Negative MSE at CSV line {reader.line_num}")
        observed = int(csv_row["observed_target_initialization_cells"])
        required_csv = int(csv_row["required_target_initialization_cells"])
        if not (
            csv_row["complete"] == "True" and saved["complete"]
            and observed == required_csv == required
            and saved["observed_initialization_target_cells"] == required
            and saved["required_initialization_target_cells"] == required
        ):
            raise ValueError(f"Incomplete evidence cell at CSV line {reader.line_num}")
        cells.append({
            "allowance_seconds": key[0], "model_id": key[1], "family": key[2],
            "noise_std": key[3], "total_labels": key[4],
            "normalized_clean_mse": error, "complete": True,
            "observed_target_initialization_cells": observed,
            "required_target_initialization_cells": required_csv,
            "source": {"id": "cells", "line": reader.line_num, "json_pointer": pointer},
        })
    if seen != expected_keys:
        raise ValueError("CSV evidence does not cover the full declared cell matrix")

    # All cells contain the same target and initialization counts, so averaging
    # these saved cell means reproduces the original hierarchical family mean.
    for summary in summaries:
        for family in axes["families"]:
            matching = [cell["normalized_clean_mse"] for cell in cells if
                        cell["model_id"] == summary["model_id"]
                        and cell["allowance_seconds"] == summary["allowance_seconds"]
                        and cell["family"] == family]
            _close(sum(matching) / len(matching), summary["family_errors"][family],
                   f"{summary['model_id']} / {family} / {summary['allowance_seconds']} s")
        balanced = sum(summary["family_errors"][family]
                       for family in axes["periodic_families"]) / len(axes["periodic_families"])
        _close(balanced, summary["balanced_periodic_error"], "balanced periodic error")

    return {
        "schema_version": 1,
        "evidence_kind": "historical_saved_result",
        "sources": sources,
        "axes": axes,
        "models": models,
        "label_splits": {
            str(total): {
                "train": int(total * tasks["train_fraction"]),
                "validation": total - int(total * tasks["train_fraction"]),
                "source": reference("/config/tasks/train_fraction"),
            }
            for total in tasks["label_budgets"]
        },
        "cells": cells,
        "summaries": summaries,
        "metric": {
            "name": "Normalized clean MSE", "units": "dimensionless", "lower_is_better": True,
            "definition": "Clean-grid mean squared error divided by clean-grid target variance",
            "source": reference("/config/analysis/primary_metric"),
        },
        "study": {
            "date": record["date"], "protocol_id": config["protocol_id"],
            "primary_time_seconds": config["training"]["primary_time_seconds"],
            "train_fraction": tasks["train_fraction"],
            "initializations_per_target": tasks["initializations"]["confirmation"],
            "targets_per_family": tasks["replicates"]["confirmation"],
            "completed_records": confirmation["completed_records"],
            "environment": confirmation["environment"],
            "aggregation": config["analysis"]["aggregation"],
            "bootstrap_unit": config["analysis"]["bootstrap_unit"],
            "scope": record["scope"],
        },
        "choices": record["choices"],
        "gate_passed": confirmation["gate_passed"],
        "gate_thresholds": config["analysis"]["primary_gate"],
        "co_primary": {
            comparator: _contrast(contrast, f"/phases/confirmation/co_primary/{comparator}")
            for comparator, contrast in confirmation["co_primary"].items()
        },
        "optimizer_contrast": _contrast(
            confirmation["optimizer_contrast"], "/phases/confirmation/optimizer_contrast"),
        "runtime_by_model": {
            model: {**runtime, "source": reference(f"/phases/confirmation/runtime_by_model/{model}")}
            for model, runtime in confirmation["runtime_by_model"].items()
        },
        "source_refs": {
            "axes": reference("/config/tasks"),
            "allowance_seconds": reference("/config/training/time_budgets_seconds"),
            "choices": reference("/choices"),
            "gate_passed": reference("/phases/confirmation/gate_passed"),
            "gate_thresholds": reference("/config/analysis/primary_gate"),
            "date": reference("/date"),
            "protocol_id": reference("/config/protocol_id"),
            "primary_time_seconds": reference("/config/training/primary_time_seconds"),
            "train_fraction": reference("/config/tasks/train_fraction"),
            "initializations_per_target": reference("/config/tasks/initializations/confirmation"),
            "targets_per_family": reference("/config/tasks/replicates/confirmation"),
            "completed_records": reference("/phases/confirmation/completed_records"),
            "environment": reference("/phases/confirmation/environment"),
            "aggregation": reference("/config/analysis/aggregation"),
            "bootstrap_unit": reference("/config/analysis/bootstrap_unit"),
            "scope": reference("/scope"),
        },
    }


def render_javascript(evidence: dict[str, Any]) -> str:
    """Serialize a self-contained, deeply immutable browser data asset."""
    payload = json.dumps(evidence, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    return (
        "// Generated by scripts/learning/build_vm_playground_evidence.py; do not hand-edit.\n"
        "// Historical saved evidence only. All sources and exact value locations are retained.\n"
        "(function () {\n  'use strict';\n"
        "  function deepFreeze(value) {\n"
        "    if (value && typeof value === 'object') {\n"
        "      Object.values(value).forEach(deepFreeze);\n"
        "      Object.freeze(value);\n"
        "    }\n    return value;\n  }\n"
        "  Object.defineProperty(window, 'VM_EVIDENCE', {\n"
        f"    value: deepFreeze({payload}),\n"
        "    writable: false, configurable: false, enumerable: true\n"
        "  });\n}());\n"
    )


def main() -> int:
    """Generate the asset, or return nonzero when the checked-in copy is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check only; write no files.")
    args = parser.parse_args()
    try:
        expected = render_javascript(build_evidence())
        output = ROOT / OUTPUT
        if args.check:
            if not output.exists() or output.read_bytes() != expected.encode("utf-8"):
                print("FAIL: evidence.js is stale or missing; source content/hashes or export changed.")
                return 1
            print("PASS: evidence.js is current; five source hashes and all 720 cells agree.")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(expected.encode("utf-8"))
            print(f"Wrote {output} ({len(expected.encode('utf-8')):,} bytes; 720 cells).")
    except (ValueError, KeyError, OSError) as error:
        print(f"FAIL: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
