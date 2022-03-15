import copy
import xarray as xr
import dacite
from vcm.data_transform import (
    qm_from_q1_q2,
    q1_from_qm_q2,
    Transform,
    TransformConfig,
    ChainedTransform,
    ChainedTransformConfig,
)

SAMPLE_DATA = xr.Dataset(
    {
        "Q1": xr.DataArray([0, 1, 2], dims=["x"]),
        "Q2": xr.DataArray([3, 4, 5], dims=["x"]),
    }
)


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
    config = TransformConfig(
        name="qm_from_q1_q2", kwargs=dict(temperature_dependent_latent_heat=True)
    )
    transform = Transform(config)
    out = transform.apply(data)
    assert "Qm" in out


def test_chained_transform():
    data = copy.deepcopy(SAMPLE_DATA)
    config_dict = {
        "transforms": [
            {"name": "vcm_derived_mapping", "kwargs": {"variables": ["Q1", "Q2"]}},
            {"name": "qm_from_q1_q2"},
        ]
    }
    transform = ChainedTransform(dacite.from_dict(ChainedTransformConfig, config_dict))
    out = transform.apply(data)
    assert "Qm" in out
