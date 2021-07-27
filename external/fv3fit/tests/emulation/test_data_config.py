import pytest
from fv3fit.emulation.data import config


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ([], slice(None)),
        ([1], slice(1)),
        ([1, 2], slice(1, 2)),
        ([1, 10, 2], slice(1, 10, 2)),
    ],
)
def test__sequence_to_slice(sequence, expected):
    result = config._sequence_to_slice(sequence)
    assert result == expected


def test__sequence_to_slice_too_long():
    with pytest.raises(ValueError):
        config._sequence_to_slice([1, 2, 3, 4])


def test__map_sequences_to_slices():
    d = {"a": [], "b": [1, 2, 2]}
    result = config._convert_map_sequences_to_slices(d)
    for k in d:
        assert k in result
        assert isinstance(result[k], slice)


def test_TransformConfig():

    transform = config.TransformConfig(
        input_variables=["a", "b"],
        output_variables=["c", "d"],
        antarctic_only=False,
        vertical_subselections={"a": slice(5, None)},
    )

    assert callable(transform)


def test_TransformConfig_from_dict():

    transform = config.TransformConfig.from_dict(
        dict(
            input_variables=["a", "b"],
            output_variables=["c", "d"],
            antarctic_only=False,
            vertical_subselections={"a": [5, None]},
        )
    )

    assert transform.vertical_subselections["a"] == slice(5, None)
    assert isinstance(transform, config.TransformConfig)
    assert callable(transform)
