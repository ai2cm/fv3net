import argparse
import pytest

from fv3fit.train_microphysics import (
    TrainConfig,
    to_flat_dict,
    to_nested_dict,
    MicrophysicsConfig,
    _get_out_samples,
    _add_items_to_parser_arguments,
    get_default_config,
)


def get_cfg_and_args_dicts():

    config_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested": {"k1": 2, "k2": 3, "double_nest": {"k1": 4, "k2": 5}},
    }

    flat_d = {
        "top": 1,
        "seq": [dict(a=1), dict(a=2)],
        "nested.k1": 2,
        "nested.k2": 3,
        "nested.double_nest.k1": 4,
        "nested.double_nest.k2": 5,
    }

    return config_d, flat_d


def test_to_flat_dict():

    config_d, expected = get_cfg_and_args_dicts()
    result = to_flat_dict(config_d)
    assert result == expected


def test_to_nested_dict():

    expected, args_d = get_cfg_and_args_dicts()
    result = to_nested_dict(args_d)
    assert result == expected


def test_flat_dict_round_trip():

    config_d, _ = get_cfg_and_args_dicts()

    args_d = to_flat_dict(config_d)
    result = to_nested_dict(args_d)

    assert result == config_d


def test__get_out_samples():

    samples = [1, 2, 3, 4]
    names = ["B", "dC", "dD", "A"]

    config = MicrophysicsConfig(
        direct_out_variables=["A", "B"],
        residual_out_variables={"C": "C_in", "D": "D_in"},
        tendency_outputs={"C": "dC", "D": "dD"},
    )

    direct, residual = _get_out_samples(config, samples, names)

    assert direct == [4, 1]
    assert residual == [2, 3]


def test__add_items_to_parser_args_no_seq():

    d = {
        "number": 1.0,
        "string.nested": "hi",
        "boolean": False,
    }

    parser = argparse.ArgumentParser()
    _add_items_to_parser_arguments(d, parser)

    default = parser.parse_args([])
    for k, v in d.items():
        parsed = vars(default)
        assert parsed[k] == v

    args = ["--number", "2.0", "--string.nested", "there"]
    specified = vars(parser.parse_args(args))
    assert specified["number"] == "2.0"
    assert specified["string.nested"] == "there"


@pytest.mark.parametrize(
    "args, expected_seq",
    [
        (["--seq"], []),
        (["--seq", "1"], ["1"]),
        (["--seq", "1", "2"], ["1", "2"]),
    ],
    ids=["empty", "single", "multiple"]
)
def test__add_items_to_parser_args_seq(args, expected_seq):

    d = {
        "seq": [1, 2, 3]
    }

    parser = argparse.ArgumentParser()
    _add_items_to_parser_arguments(d, parser)

    parsed = parser.parse_args(args)
    assert parsed.seq == expected_seq


def test__add_items_to_parser_args_mapping_error():

    d = {"mapping": dict(a=1)}
    parser = argparse.ArgumentParser()

    with pytest.raises(ValueError):
        _add_items_to_parser_arguments(d, parser)


def test_get_default_config():

    config = get_default_config()
    assert isinstance(config, TrainConfig)
