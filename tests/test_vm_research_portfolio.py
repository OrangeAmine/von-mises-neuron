"""Protect the completed-study export against stale or overstated evidence."""

import copy
import hashlib
import json

import pytest

from scripts.learning.build_vm_research_portfolio import (
    OUTPUT, ROOT, SPEC, _number, build_portfolio, render_javascript, resolve_pointer,
)


@pytest.fixture(scope="module")
def portfolio():
    return build_portfolio()


def test_export_is_current_and_preserves_source_values(portfolio):
    assert (ROOT / OUTPUT).read_bytes() == render_javascript(portfolio).encode("utf-8")
    for study in portfolio["studies"]:
        for view in study["views"]:
            for metric in view["metrics"]:
                raw = (ROOT / portfolio["sources"][metric["source"]]["path"]).read_bytes()
                record = json.loads(raw)
                assert metric["value"] == resolve_pointer(record, metric["pointer"])
                if metric["interval_pointer"]:
                    assert metric["interval"] == resolve_pointer(record, metric["interval_pointer"])


def test_scientific_failures_are_not_replaced_by_completion(portfolio):
    studies = {s["id"]: s for s in portfolio["studies"]}
    for key in ("N01", "N04", "N05", "N06"):
        assert studies[key]["gate"]["passed"] is False
        assert studies[key]["status"] == "Overall scientific gate failed"
    assert studies["N02"]["gate"] is None
    assert studies["N03"]["gate"] is None


def test_primary_and_conditional_uncertainty_remain_distinct(portfolio):
    study = next(s for s in portfolio["studies"] if s["id"] == "N05")
    primary, diagnostic = [v["metrics"][0] for v in study["views"]]
    assert primary["value"] == diagnostic["value"]
    assert primary["interval"][0] < 0 < primary["interval"][1]
    assert diagnostic["interval"][0] > 0
    assert primary["interval_pointer"] != diagnostic["interval_pointer"]


def test_objective_change_retains_the_reactive_control_win(portfolio):
    study = next(s for s in portfolio["studies"] if s["id"] == "N06")
    low, high = study["views"]
    assert low["metrics"][0]["value"] < low["metrics"][2]["value"]
    assert high["metrics"][0]["value"] > high["metrics"][2]["value"]


def test_changed_evidence_is_rejected(tmp_path):
    spec = json.loads((ROOT / SPEC).read_text(encoding="utf-8"))
    # Fail on the first pinned source before later links need to be copied.
    source = next(iter(spec["sources"].values()))
    path = tmp_path / source["path"]
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
    target = tmp_path / SPEC
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="Pinned evidence changed"):
        build_portfolio(tmp_path)


@pytest.mark.parametrize("value", [None, True, float("nan"), float("inf")])
def test_non_numeric_and_nonfinite_evidence_is_rejected(value):
    with pytest.raises(ValueError, match="Invalid numeric"):
        _number(value)


def test_pointer_supports_arrays_and_escaped_keys():
    assert resolve_pointer({"a/b": [{"~key": 4}]}, "/a~1b/0/~0key") == 4
