import pytest
import tempfile
from fv3fit.train_microphysics import (
    _update_TrainConfig_model_selection,
    TrainConfig
)


def test_model_selection_update():


    d = {
        "model": {
            "selection_map": {
                "key1": [1, 3],
            },
            "unedited_key": 2
        }
    }

    result = _update_TrainConfig_model_selection(d)
    assert d["model"]["selection_map"]
    assert isinstance(d["model"]["selection_map"]["key1"], slice)
    assert d["model"]["unedited_key"] == 2

