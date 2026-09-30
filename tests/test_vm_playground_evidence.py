"""Focused checks for the immutable offline teaching evidence export."""

import copy
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path

import pytest

from scripts.learning.build_vm_playground_evidence import (
    OUTPUT, ROOT, SOURCE_PATHS, SOURCE_SHA256, build_evidence, render_javascript,
)


@pytest.fixture(scope="module")
def evidence():
    """Build evidence once from the unchanged historical documentation."""
    return build_evidence()


def test_export_is_current_and_source_hashes_match(evidence):
    assert (ROOT / OUTPUT).read_bytes() == render_javascript(evidence).encode("utf-8")
    for source in evidence["sources"].values():
        assert source["sha256"] == hashlib.sha256((ROOT / source["path"]).read_bytes()).hexdigest()
        assert (ROOT / OUTPUT.parent / source["href"]).resolve() == (ROOT / source["path"]).resolve()


def test_every_model_family_noise_label_and_allowance_is_retained(evidence):
    assert len(evidence["models"]) == 10
    assert len(evidence["cells"]) == 720
    assert len(evidence["summaries"]) == 30
    assert set(evidence["axes"]["families"]) == {"smooth", "narrow", "multimodal", "nonperiodic"}
    assert evidence["axes"]["allowance_seconds"] == [0.25, 0.75, 2.0]
    assert evidence["axes"]["total_labels"] == [64, 128, 512]
    assert set(evidence["runtime_by_model"]) == {model["id"] for model in evidence["models"]}
    assert [(item["train"], item["validation"]) for item in evidence["label_splits"].values()] == [
        (48, 16), (96, 32), (384, 128)]


def test_every_csv_value_has_exact_dual_source_locations(evidence):
    record = json.loads((ROOT / SOURCE_PATHS["record"]).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(io.StringIO((ROOT / SOURCE_PATHS["cells"]).read_text(encoding="utf-8"))))
    for cell in evidence["cells"]:
        csv_row = rows[cell["source"]["line"] - 2]
        saved = record
        for segment in cell["source"]["json_pointer"].split("/")[1:]:
            saved = saved[int(segment)] if isinstance(saved, list) else saved[segment]
        assert cell["normalized_clean_mse"] == float(csv_row["normalized_clean_mse"]) == saved["mean_error"]
        assert cell["observed_target_initialization_cells"] == 16


def test_failed_gate_and_unfavorable_optimizer_contrast_are_preserved(evidence):
    assert evidence["gate_passed"] is False
    fixed16 = evidence["co_primary"]["fixed16"]
    assert fixed16["relative_improvement"] > 0.54
    assert fixed16["family_improvements"]["smooth"] < -0.47
    assert fixed16["gate_passed"] is False
    assert evidence["co_primary"]["fixed48"]["gate_passed"] is True
    assert evidence["optimizer_contrast"]["relative_improvement"] < -0.15
    assert evidence["gate_thresholds"]["each_comparison_max_family_regression"] == 0.1


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "changed_value", "incomplete"])
def test_inconsistent_csv_is_rejected(tmp_path: Path, corruption: str, monkeypatch):
    for relative in SOURCE_PATHS.values():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    cells_path = tmp_path / SOURCE_PATHS["cells"]
    lines = cells_path.read_text(encoding="utf-8").splitlines()
    if corruption == "duplicate":
        lines.append(lines[1])
    elif corruption == "missing":
        lines.pop()
    else:
        row = lines[1].split(",")
        row[5 if corruption == "changed_value" else 6] = "0.9" if corruption == "changed_value" else "False"
        lines[1] = ",".join(row)
    cells_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Isolate semantic corruption guards from the separate immutable-hash guard.
    monkeypatch.setitem(SOURCE_SHA256, "cells", hashlib.sha256(cells_path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        build_evidence(tmp_path)


def test_changed_snapshot_source_is_rejected(monkeypatch):
    monkeypatch.setitem(SOURCE_SHA256, "record", "0" * 64)
    with pytest.raises(ValueError, match="Historical source differs from inherited snapshot"):
        build_evidence()


def test_generated_javascript_declares_nonwritable_deeply_frozen_data(evidence):
    # Browser mutation behavior is checked by the UI inspection; this also guards
    # against accidentally switching this historical asset into mutable app state.
    output = render_javascript(copy.deepcopy(evidence))
    assert "Object.values(value).forEach(deepFreeze)" in output
    assert "Object.freeze(value)" in output
    assert "writable: false, configurable: false" in output
