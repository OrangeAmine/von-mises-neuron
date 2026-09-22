"""Storage repair tests: preserve numerical values and reject invalid JSON."""

import json

import numpy as np
import pytest

from scripts.run_vm_n04_v1_1 import write_json


def test_numpy_metadata_serialization_does_not_change_values(tmp_path):
    path = tmp_path / "recovered.json"
    value = {"irregular_start_steps": [np.int32(31), np.int64(143)],
             "metric": np.float32(.5), "selection": np.bool_(True)}
    write_json(path, value)
    assert json.loads(path.read_text()) == {
        "irregular_start_steps": [31,143], "metric": .5, "selection": True}


@pytest.mark.parametrize("value", [np.float32(np.nan), np.float64(np.inf), float("nan")])
def test_repair_does_not_hide_nonfinite_numbers(tmp_path, value):
    path = tmp_path / "failed.json"
    with pytest.raises(ValueError):
        write_json(path, {"metric": value})
    assert not path.exists()


def test_repair_rejects_unknown_objects_without_truncating_existing_file(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text('{"original":true}')
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})
    assert path.read_text() == '{"original":true}'
