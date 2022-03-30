import xarray as xr
import dacite
from vcm.data_transform import DATA_TRANSFORM_REGISTRY
import vcm
import pytest

ARRAY = xr.DataArray([0, 1, 2], dims=["x"])


def test_all_registered_transforms_are_added_to_data_transform_name_type():
    for key in DATA_TRANSFORM_REGISTRY:
        try:
            dacite.from_dict(vcm.DataTransform, {"name": key})
        except dacite.exceptions.WrongTypeError as error:
            raise NotImplementedError(
                "Newly registered transforms must be added to the "
                "vcm.data_transform.TransformName type."
            ) from error


def test_all_transform_functions():
    for key in DATA_TRANSFORM_REGISTRY:
        input_variables = DATA_TRANSFORM_REGISTRY[key].inputs
        data = xr.Dataset({name: ARRAY for name in input_variables})
        out = DATA_TRANSFORM_REGISTRY[key].func(data)
        for output_name in DATA_TRANSFORM_REGISTRY[key].outputs:
            assert output_name in out


def test_data_transform():
    data = xr.Dataset({"Q1": ARRAY, "Q2": ARRAY})
    transform = vcm.DataTransform(name="Qm_from_Q1_Q2")
    out = transform.apply(data)
    assert "Qm" in out


def test_chained_data_transform():
    data = xr.Dataset({"dQ1": ARRAY, "pQ1": ARRAY, "Q2": ARRAY})
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
