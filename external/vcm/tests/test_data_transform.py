import copy
import vcm
import pytest
import xarray as xr

SAMPLE_DATA = {
    "Q1": xr.DataArray([0, 1, 2], dims=["x"]),
    "Q2": xr.DataArray([3, 4, 5], dims=["x"]),
}


@pytest.mark.parametrize(
    "variable_names, expected_transform",
    [
        ([], vcm.IdentityDataTransform),
        (["air_temperature"], vcm.IdentityDataTransform),
        (["Q1", "Q2"], vcm.QmDataTransform),
        (["Q1", "Q2"], vcm.QmDataTransform),
        (["Qm", "Q2"], vcm.QmDataTransform),
        (["Q1", "Q2", "air_temperature"], vcm.QmDataTransform),
    ],
)
def test_detect_transform(variable_names, expected_transform):
    transform = vcm.detect_transform(variable_names)
    assert isinstance(transform, expected_transform)


@pytest.mark.parametrize(
    "transform", [vcm.IdentityDataTransform(), vcm.QmDataTransform()]
)
def test_round_trip(transform):
    data_copy = copy.deepcopy(SAMPLE_DATA)
    out = transform.backward(transform.forward(data_copy))
    for name in SAMPLE_DATA:
        xr.testing.assert_allclose(SAMPLE_DATA[name], out[name])


def test_qm_transform_forward():
    out = vcm.QmDataTransform().forward(copy.deepcopy(SAMPLE_DATA))
    assert "Qm" in out


def test_qm_transform_backward():
    data = copy.deepcopy(SAMPLE_DATA)
    data["Qm"] = data.pop("Q1")
    out = vcm.QmDataTransform().backward(data)
    assert "Q1" in out


def test_qm_transform_backward_no_transform_needed():
    data = copy.deepcopy(SAMPLE_DATA)
    out = vcm.QmDataTransform().backward(data)
    assert "Q1" in out
