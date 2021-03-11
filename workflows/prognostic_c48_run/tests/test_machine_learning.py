from runtime.steppers.machine_learning import PureMLStepper, MLStateStepper
from machine_learning_mocks import get_mock_sklearn_model, get_mock_rad_flux_model
import requests
import xarray as xr
import joblib
import numpy as np
import yaml
import pytest


def checksum_xarray(xobj):
    return joblib.hash(np.asarray(xobj))


def checksum_xarray_dict(d):
    sorted_keys = sorted(d.keys())
    return [(key, checksum_xarray(d[key])) for key in sorted_keys]


@pytest.fixture(scope="session")
def state(tmp_path_factory):
    url = "https://github.com/VulcanClimateModeling/vcm-ml-example-data/blob/b100177accfcdebff2546a396d2811e32c01c429/fv3net/prognostic_run/inputs_4x4.nc?raw=true"  # noqa
    r = requests.get(url)
    lpath = tmp_path_factory.getbasetemp() / "input_data.nc"
    lpath.write_bytes(r.content)
    return xr.open_dataset(str(lpath))


@pytest.fixture(params=["PureMLStepper", "MLStateStepper"])
def ml_stepper_name(request):
    return request.param


@pytest.fixture
def mock_model(ml_stepper_name):
    if ml_stepper_name == "PureMLStepper":
        model = get_mock_sklearn_model()
    elif ml_stepper_name == "MLStateStepper":
        model = get_mock_rad_flux_model()
    else:
        raise ValueError("ML Stepper name not defined.")
    return model


@pytest.fixture
def ml_stepper(ml_stepper_name, mock_model):
    timestep = 900
    if ml_stepper_name == "PureMLStepper":
        stepper = PureMLStepper(mock_model, timestep)
    elif ml_stepper_name == "MLStateStepper":
        stepper = MLStateStepper(mock_model, timestep)
    return stepper


def test_MLStepper_schema_unchanged(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    xr.Dataset(diagnostics).info(regtest)
    xr.Dataset(tendencies).info(regtest)
    xr.Dataset(states).info(regtest)


def test_state_regression(state, regtest):
    checksum = checksum_xarray_dict(state)
    print(checksum, file=regtest)


def test_MLStepper_regression_checksum(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    checksums = yaml.safe_dump(
        [
            ("tendencies", checksum_xarray_dict(tendencies)),
            ("diagnostics", checksum_xarray_dict(diagnostics)),
            ("states", checksum_xarray_dict(states)),
        ]
    )

    print(checksums, file=regtest)
