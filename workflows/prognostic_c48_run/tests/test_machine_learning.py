from runtime.steppers.machine_learning import PureMLStepper
from machine_learning_mocks import get_mock_predictor
import requests
import xarray as xr
import yaml
import pytest
import vcm.testing


@pytest.fixture(scope="session")
def state(tmp_path_factory):
    url = "https://github.com/VulcanClimateModeling/vcm-ml-example-data/blob/b100177accfcdebff2546a396d2811e32c01c429/fv3net/prognostic_run/inputs_4x4.nc?raw=true"  # noqa
    r = requests.get(url)
    lpath = tmp_path_factory.getbasetemp() / "input_data.nc"
    lpath.write_bytes(r.content)
    return xr.open_dataset(str(lpath))


@pytest.fixture(params=["tendencies", "rad_fluxes"])
def model_output_type(request):
    return request.param


@pytest.fixture
def ml_stepper(model_output_type):
    timestep = 900
    mock_model = get_mock_predictor(model_output_type)
    ml_stepper = PureMLStepper(mock_model, timestep, hydrostatic=False)
    return ml_stepper


@pytest.mark.parametrize("output_type", ["tendencies", "rad_fluxes"])
def test_mock_predictor_checksum(output_type, state, regtest):
    mock_model = get_mock_predictor(output_type)
    predicted = mock_model.predict(state)
    test_state_regression(predicted, regtest)


def test_ml_steppers_schema_unchanged(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    xr.Dataset(diagnostics).info(regtest)
    xr.Dataset(tendencies).info(regtest)
    xr.Dataset(states).info(regtest)


def test_state_regression(state, regtest):
    checksum = vcm.testing.checksum_dataarray_mapping(state)
    print(checksum, file=regtest)


def test_ml_steppers_regression_checksum(state, ml_stepper, regtest):
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    checksums = yaml.safe_dump(
        [
            ("tendencies", vcm.testing.checksum_dataarray_mapping(tendencies)),
            ("diagnostics", vcm.testing.checksum_dataarray_mapping(diagnostics)),
            ("states", vcm.testing.checksum_dataarray_mapping(states)),
        ]
    )

    print(checksums, file=regtest)


@pytest.mark.parametrize(
    "mock_predictor", ["state_and_tendency", "state_tendency_and_intermediate"]
)
def test_ml_stepper_state_update(state, mock_predictor):
    ml_stepper = PureMLStepper(
        get_mock_predictor(mock_predictor), timestep=900, hydrostatic=False
    )
    (tendencies, diagnostics, states) = ml_stepper(None, state)
    assert set(states) == {"total_sky_downward_shortwave_flux_at_surface"}
    assert set(tendencies) == {"dQ1"}
    if mock_predictor == "state_tendendency_and_intermediate":
        assert "shortwave_transmissivity_of_atmospheric_column" in diagnostics
