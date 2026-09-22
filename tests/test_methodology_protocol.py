"""Contract counts and fail-closed confirmation provenance tests."""

import copy
from pathlib import Path

import pytest

import src.methodology_protocol as protocol


@pytest.fixture
def config():
    return protocol.read_json(protocol.CONFIG_PATH)


@pytest.fixture
def choices(config):
    return {"A": {"Adam": 0.01, "SGD": 0.03}, "B": {
        model["id"]: 0.001 for model in config["B"]["models"] if model["kind"] == "gradient"}}


@pytest.mark.parametrize("stage,phase,gradient,ridge,reference", [
    ("A","pilot",48,0,4), ("A","development",64,0,16),
    ("A","confirmation",256,0,64), ("B","pilot",32,8,0),
    ("B","development",192,48,0), ("B","confirmation",576,144,0)])
def test_exact_matrix_counts(config,choices,stage,phase,gradient,ridge,reference):
    rows = protocol.planned_runs(config,stage,phase,choices)
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert all(row["status"] == "pending" for row in rows)
    assert [sum(row["kind"] == kind for row in rows) for kind in ("gradient","ridge","reference")] == [gradient,ridge,reference]


def test_missing_lock_never_opens_confirmation(config,tmp_path):
    with pytest.raises(ValueError,match="lock"):
        protocol.verify_confirmation_lock(None,config)
    with pytest.raises(ValueError,match="missing"):
        protocol.verify_confirmation_lock(tmp_path/"absent.json",config)


def locked_fixture(tmp_path,config,choices):
    for name in protocol.SOURCE_PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("fixture only; no generator execution",encoding="utf-8")
    development = tmp_path/"results/development.json"
    protocol.write_json(development,{"status":"completed"})
    lock_path = tmp_path/"lock.json"
    protocol.write_confirmation_lock(lock_path,config,choices,"fourier_ridge",[development],repo_root=tmp_path)
    return lock_path,development


@pytest.mark.parametrize("change",["source","config","development","lock"])
def test_changed_dependencies_rejected(config,choices,tmp_path,change):
    path,development = locked_fixture(tmp_path,config,choices)
    assert protocol.verify_confirmation_lock(path,config,repo_root=tmp_path)["B_primary_comparator"] == "fourier_ridge"
    if change == "source":
        (tmp_path/protocol.SOURCE_PATHS[0]).write_text("changed")
    elif change == "development":
        development.write_text("{}")
    elif change == "config":
        config = copy.deepcopy(config)
        config["A"]["updates"] = 1
    else:
        path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError,match="changed|Changed"):
        protocol.verify_confirmation_lock(path,config,repo_root=tmp_path)


def test_json_rejects_nonfinite_and_overwrites(tmp_path):
    path = tmp_path/"one.json"
    with pytest.raises(ValueError):
        protocol.write_json(path,{"metric":float("nan")})
    protocol.write_json(path,{"ok":True})
    with pytest.raises(FileExistsError):
        protocol.write_json(path,{"ok":False})
