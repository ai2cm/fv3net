import pytest
import tempfile
from fv3fit.train_microphysics import (
    TrainConfig,
    config_dict_to_flat_args_dict,
    args_dict_to_config_dict,
)


def get_cfg_and_args_dicts():

    config_d = {
        "top": 1,
        "nested": {"k1": 2, "k2": 3, "double_nest": {"k1": 4, "k2": 5,}},
    }

    flat_d = {
        "top": 1,
        "nested.k1": 2,
        "nested.k2": 3,
        "nested.double_nest.k1": 4,
        "nested.double_nest.k2": 5,
    }

    return config_d, flat_d


def test_config_dict_to_flat_args_dict():

    config_d, expected = get_cfg_and_args_dicts()
    result = config_dict_to_flat_args_dict(config_d)
    assert result == expected


def test_args_dict_to_config_dict():

    expected, args_d = get_cfg_and_args_dicts()
    result = args_dict_to_config_dict(args_d)
    assert result == expected


def test_flat_dict_round_trip():

    config_d, _ = get_cfg_and_args_dicts()

    args_d = config_dict_to_flat_args_dict(config_d)
    result = args_dict_to_config_dict(args_d)

    assert result == config_d
