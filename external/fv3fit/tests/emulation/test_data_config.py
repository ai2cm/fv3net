import pytest
from fv3fit.emulation.data import config
from fv3fit.emulation.data.config import SliceConfig, TransformConfig


@pytest.mark.parametrize("start", [None, 1])
@pytest.mark.parametrize("stop", [None, 5])
@pytest.mark.parametrize("step", [None, 2])
def test_SliceConfig(start, stop, step):
    expected = slice(start, stop, step)
    config = SliceConfig(start=start, stop=stop, step=step)
    assert config.slice == expected


def test_TransformConfig():

    transform = config.TransformConfig(
        input_variables=["a", "b"],
        output_variables=["c", "d"],
        antarctic_only=False,
        vertical_subselections={"a": SliceConfig(start=5)},
    )

    assert transform.vert_sel_as_slices["a"] == slice(5, None)
    assert callable(transform)


def test_TransformConfig_from_dict():

    transform = config.TransformConfig.from_dict(
        dict(
            input_variables=["a", "b"],
            output_variables=["c", "d"],
            antarctic_only=False,
            vertical_subselections={"a": dict(start=5)},
        )
    )

    assert transform.vert_sel_as_slices["a"] == slice(5, None)
    assert isinstance(transform, config.TransformConfig)
    assert callable(transform)
