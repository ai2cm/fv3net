import copy
import xarray as xr
import dacite
from vcm.data_transform import qm_from_q1_q2, q1_from_qm_q2, DATA_TRANSFORM_REGISTRY
import vcm

ARRAY = xr.DataArray([0, 1, 2], dims=["x"])


def test_all_registered_transforms_are_added_to_data_transform_name_type():
    for key in DATA_TRANSFORM_REGISTRY:
        try:
            dacite.from_dict(vcm.DataTransformConfig, {"name": key})
        except dacite.exceptions.WrongTypeError as error:
            raise NotImplementedError(
                "Newly registered transforms must be added to the "
                "vcm.data_transform.TransformName type."
            ) from error


def test_all_transform_functions():
    for key in DATA_TRANSFORM_REGISTRY:
        input_variables = DATA_TRANSFORM_REGISTRY[key]["inputs"]
        data = xr.Dataset({name: ARRAY for name in input_variables})
        out = DATA_TRANSFORM_REGISTRY[key]["func"](data)
        for output_name in DATA_TRANSFORM_REGISTRY[key]["outputs"]:
            assert output_name in out


def test_data_transform():
    data = xr.Dataset({"Q1": ARRAY, "Q2": ARRAY, "air_temperature": ARRAY})
    config = vcm.DataTransformConfig(
        name="qm_from_q1_q2", kwargs=dict(temperature_dependent_latent_heat=True)
    )
    transform = vcm.DataTransform(config)
    out = transform.apply(data)
    assert "Qm" in out


def test_chained_data_transform():
    data = xr.Dataset({"dQ1": ARRAY, "pQ1": ARRAY, "Q2": ARRAY})
    data["dQ1"] = data.pQ1
    config_dict = {"config": [{"name": "q1_from_dQ1_pQ1"}, {"name": "qm_from_q1_q2"}]}
    transform = dacite.from_dict(vcm.ChainedDataTransform, config_dict)
    out = transform.apply(data)
    assert "Qm" in out


def test_transform_inputs_outputs():
    transform = vcm.DataTransform(vcm.DataTransformConfig("qm_from_q1_q2"))
    assert transform.input_variables == ["Q1", "Q2"]
    assert transform.output_variables == ["Qm"]


def test_chained_transform_inputs_outputs():
    config_dict = {"config": [{"name": "q1_from_dQ1_pQ1"}, {"name": "qm_from_q1_q2"}]}
    transform = dacite.from_dict(vcm.ChainedDataTransform, config_dict)
    assert transform.input_variables == ["Q2", "dQ1", "pQ1"]
    assert transform.output_variables == ["Q1", "Qm"]
