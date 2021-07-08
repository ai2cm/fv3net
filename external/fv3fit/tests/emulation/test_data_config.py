import pytest
from fv3fit.emulation.data import config


@pytest.mark.parametrize(
    "sequence, expected",
    [
        ([], slice(None)),
        ([1], slice(1)),
        ([1, 2], slice(1, 2)),
        ([1, 10, 2], slice(1, 10, 2))
    ]
)
def test__sequence_to_slice(sequence, expected):
    result = config._sequence_to_slice(sequence)
    assert result == expected


def test__sequence_to_slice_too_long():
    with pytest.raises(ValueError):
        config._sequence_to_slice([1, 2, 3, 4])


def test_TransformConfig():

    result = config.TransformConfig(
        input_variables=["a", "b"],
        output_variables=["c", "d"],
        antarctic_only=False,
        vertical_subselections={"a": slice(5, None)},
    )

    transform_func = result.get_transform_pipeline()
    assert callable(transform_func)


def test_TransformConfig_from_dict():

    result = config.TransformConfig.from_dict(
        dict(
            input_variables=["a", "b"],
            output_variables=["c", "d"],
            antarctic_only=False,
            vertical_subselections={"a": [5, None]},
        )
    )

    assert isinstance(result, config.TransformConfig)
    transform_func = result.get_transform_pipeline()
    assert callable(transform_func)
