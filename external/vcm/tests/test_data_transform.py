import numpy as np
import xarray as xr
import dacite
from vcm.data_transform import (
    DATA_TRANSFORM_REGISTRY,
    DLW_SFC,
    DSW_SFC,
    DSW_TOA,
    ULW_SFC,
    ULW_TOA,
    USW_SFC,
    USW_TOA,
    LHF,
    SHF,
    PRECIP,
    COL_T_NUDGE,
)
import vcm
import pytest

# to construct datasets that operate on all transforms, specify which
# variables are 2D here. All others are assumed to be 3D.
VARIABLES_2D = {
    DLW_SFC,
    DSW_SFC,
    DSW_TOA,
    ULW_SFC,
    ULW_TOA,
    USW_SFC,
    USW_TOA,
    LHF,
    SHF,
    PRECIP,
    COL_T_NUDGE,
    "implied_downward_radiative_flux_at_surface",
    "implied_surface_precipitation_rate",
}


def get_3d_array():
    return xr.DataArray(np.random.random_sample((6, 4, 4)), dims=["z", "y", "x"])


def get_2d_array():
    return xr.DataArray(np.random.random_sample((4, 4)), dims=["y", "x"])


def get_dataset(names):
    return xr.Dataset(
        {
            name: get_2d_array() if name in VARIABLES_2D else get_3d_array()
            for name in names
        }
    )


def test_all_registered_transforms_are_added_to_data_transform_name_type():
    for key in DATA_TRANSFORM_REGISTRY:
        try:
            dacite.from_dict(vcm.DataTransform, {"name": key})
        except dacite.exceptions.WrongTypeError as error:
            raise NotImplementedError(
                "Newly registered transforms must be added to the "
                "vcm.data_transform.TransformName type."
            ) from error


@pytest.mark.parametrize("key", list(DATA_TRANSFORM_REGISTRY))
def test_transform_correctly_specify_inputs_and_outputs(key):
    data = get_dataset(DATA_TRANSFORM_REGISTRY[key].inputs)
    out = DATA_TRANSFORM_REGISTRY[key].func(data)
    for output_name in DATA_TRANSFORM_REGISTRY[key].outputs:
        assert output_name in out


def test_data_transform():
    data = get_dataset(["Q1", "Q2"])
    transform = vcm.DataTransform(name="Qm_from_Q1_Q2")
    out = transform.apply(data)
    assert "Qm" in out


def test_chained_data_transform():
    data = get_dataset(["dQ1", "pQ1", "Q2"])
    data["dQ1"] = data.pQ1
    config_dict = {
        "transforms": [{"name": "Q1_from_dQ1_pQ1"}, {"name": "Qm_from_Q1_Q2"}]
    }
    transform = dacite.from_dict(vcm.ChainedDataTransform, config_dict)
    out = transform.apply(data)
    assert "Qm" in out


def test_transform_inputs_outputs():
    transform = vcm.DataTransform("Qm_from_Q1_Q2")
    assert transform.input_variables == ["Q1", "Q2"]
    assert transform.output_variables == ["Qm"]


@pytest.mark.parametrize(
    "transforms, expected_inputs, expected_outputs",
    [
        ([], [], [],),
        (
            [{"name": "Q1_from_dQ1_pQ1"}, {"name": "Qm_from_Q1_Q2"}],
            ["Q2", "dQ1", "pQ1"],
            ["Q1", "Qm"],
        ),
        (
            [{"name": "Q1_from_dQ1_pQ1"}, {"name": "Q2_from_dQ2_pQ2"}],
            ["dQ1", "dQ2", "pQ1", "pQ2"],
            ["Q1", "Q2"],
        ),
    ],
)
def test_chained_transform_inputs_outputs(
    transforms, expected_inputs, expected_outputs
):
    config_dict = {"transforms": transforms}
    transform = dacite.from_dict(vcm.ChainedDataTransform, config_dict)
    assert transform.input_variables == expected_inputs
    assert transform.output_variables == expected_outputs


@pytest.mark.parametrize(
    "name, kwargs",
    [
        ("Qm", {"rectify_downward_radiative_flux": False}),
        ("Q2", {"rectify_surface_precipitation_rate": False}),
    ],
)
def test_flux_transform_round_trip(name, kwargs):
    forward_name = f"{name}_flux_from_{name}_tendency"
    backward_name = f"{name}_tendency_from_{name}_flux"
    data = get_dataset(DATA_TRANSFORM_REGISTRY[forward_name].inputs)
    data_with_flux = DATA_TRANSFORM_REGISTRY[forward_name].func(data, **kwargs)
    data_with_flux = data_with_flux.drop_vars([name])
    round_tripped = DATA_TRANSFORM_REGISTRY[backward_name].func(data_with_flux)
    xr.testing.assert_allclose(data[name], round_tripped[name])
