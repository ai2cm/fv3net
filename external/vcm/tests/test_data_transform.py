import copy
import xarray as xr
import dacite
from vcm.data_transform import qm_from_q1_q2, q1_from_qm_q2, DATA_TRANSFORM_FUNCTIONS
import vcm

SAMPLE_DATA = xr.Dataset(
    {
        "Q1": xr.DataArray([0, 1, 2], dims=["x"]),
        "Q2": xr.DataArray([3, 4, 5], dims=["x"]),
    }
)


def test_all_registered_transforms_are_added_to_data_transform_name_type():
    for key in DATA_TRANSFORM_FUNCTIONS:
        try:
            dacite.from_dict(vcm.DataTransformConfig, {"name": key})
        except dacite.exceptions.WrongTypeError as error:
            raise NotImplementedError(
                "Newly registered transforms must be added to the TransformName type."
            ) from error


def test_qm_from_q1_q2_transform():
    data = copy.deepcopy(SAMPLE_DATA)
    out = qm_from_q1_q2(data)
    assert "Qm" in out


def test_q1_from_qm_q2_transform():
    data = copy.deepcopy(SAMPLE_DATA)
    data = data.rename(Q1="Qm")
    out = q1_from_qm_q2(data)
    assert "Q1" in out


def test_transform():
    data = copy.deepcopy(SAMPLE_DATA)
    data["air_temperature"] = data["Q1"]
    config = vcm.DataTransformConfig(
        name="qm_from_q1_q2", kwargs=dict(temperature_dependent_latent_heat=True)
    )
    transform = vcm.DataTransform(config)
    out = transform.apply(data)
    assert "Qm" in out


def test_chained_transform():
    data = copy.deepcopy(SAMPLE_DATA)
    data = data.rename(Q1="pQ1")
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
