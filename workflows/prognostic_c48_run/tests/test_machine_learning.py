from runtime.steppers.machine_learning import PureMLStepper
from machine_learning_mocks import get_mock_sklearn_model
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
