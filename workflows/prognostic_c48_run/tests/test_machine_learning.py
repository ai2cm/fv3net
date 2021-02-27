from runtime.steppers.machine_learning import PureMLStepper, EmulatorStepper
from machine_learning_mocks import get_mock_sklearn_model, get_emulator
import xarray as xr
from pathlib import Path
import joblib
import numpy as np
import yaml
import pytest
import vcm.derived_mapping
import cftime


def checksum_xarray(xobj):
    return joblib.hash(np.asarray(xobj))


def checksum_xarray_dict(d):
    sorted_keys = sorted(d.keys())
    return [(key, checksum_xarray(d[key])) for key in sorted_keys]


@pytest.fixture()
def state():
    INPUT_DATA = Path(__file__).parent / "input_data" / "inputs_4x4.nc"
    ds = xr.open_dataset(INPUT_DATA.as_posix())
    return vcm.derived_mapping.DerivedMapping(ds)


@pytest.fixture(params=["ml", "emulator"])
def stepper(request):
    timestep = 900
    if request.param == "ml":
        model = get_mock_sklearn_model()
        return PureMLStepper(model, timestep)
    elif request.param == "emulator":
        return EmulatorStepper(get_emulator(), timestep)


def test_PureMLStepper_schema_unchanged(state, regtest, stepper):
    time = cftime.DatetimeJulian(2016, 8, 1)
    (tendencies, diagnostics, state_updates) = stepper(time, state)
    xr.Dataset(diagnostics).info(regtest)
    xr.Dataset(tendencies).info(regtest)
    xr.Dataset(state_updates).info(regtest)


def test_state_regression(state, regtest):
    checksum = checksum_xarray_dict(state)
    print(checksum, file=regtest)


def test_PureMLStepper_regression_checksum(state, regtest, stepper):
    time = cftime.DatetimeJulian(2016, 8, 1)
    (tendencies, diagnostics, state_updates) = stepper(time, state)
    checksums = yaml.safe_dump(
        [
            ("tendencies", checksum_xarray_dict(tendencies)),
            ("diagnostics", checksum_xarray_dict(diagnostics)),
            ("state_updates", checksum_xarray_dict(state_updates)),
        ]
    )

    print(checksums, file=regtest)
