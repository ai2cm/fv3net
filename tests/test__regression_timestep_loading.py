import pytest
from fv3net.regression.sklearn.__main__ import _timesteps_to_list

TRAIN_TIMESTEPS = range(10)


def test__timesteps_to_list():
    timestep_config = {
        "train": [i for i in TRAIN_TIMESTEPS],
        "test": [-1 for i in range(10)],
    }
    assert set(_timesteps_to_list(timestep_config)) == set(range(10))


def test__timesteps_to_list_paired():
    timestep_config = {
        "train": [[i, -1] for i in TRAIN_TIMESTEPS],
        "test": [-1 for i in range(10)],
    }
    assert set(_timesteps_to_list(timestep_config)) == set(TRAIN_TIMESTEPS)


def test__timesteps_to_list_invalid_input():
    invalid_configs = [
        {"test": [[i, -i] for i in TRAIN_TIMESTEPS]},
        {"train": []},
    ]
    for invalid_config in invalid_configs:
        with pytest.raises(Exception):
            _timesteps_to_list(invalid_config)
