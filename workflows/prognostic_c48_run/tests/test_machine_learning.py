from runtime.steppers.machine_learning import PureMLStepper
from machine_learning_mocks import get_mock_sklearn_model
import xarray as xr
from pathlib import Path
import joblib
import numpy as np
import yaml
import pytest


def checksum_xarray(xobj):
    return joblib.hash(np.asarray(xobj))


def checksum_xarray_dict(d):
    sorted_keys = sorted(d.keys())
    return [(key, checksum_xarray(d[key])) for key in sorted_keys]


@pytest.fixture()
def state():
    INPUT_DATA = Path(__file__).parent / "input_data" / "inputs_4x4.nc"
    return xr.open_dataset(INPUT_DATA.as_posix())


def test_PureMLStepper_schema_unchanged(state, regtest):
    model = get_mock_sklearn_model()
    timestep = 900
    (tendencies, diagnostics, _,) = PureMLStepper(model, timestep)(None, state)
    xr.Dataset(diagnostics).info(regtest)
    xr.Dataset(tendencies).info(regtest)


def test_state_regression(state, regtest):
    checksum = checksum_xarray_dict(state)
    print(checksum, file=regtest)


def test_PureMLStepper_regression_checksum(state, regtest):
    model = get_mock_sklearn_model()
    timestep = 900
    (tendencies, diagnostics, _,) = PureMLStepper(model, timestep)(None, state)
    checksums = yaml.safe_dump(
        [
            ("tendencies", checksum_xarray_dict(tendencies)),
            ("diagnostics", checksum_xarray_dict(diagnostics)),
        ]
    )

    print(checksums, file=regtest)
