import joblib
import fv3fit
import vcm
import numpy as np
import xarray as xr
import pytest
from runtime.transformers.fv3fit import (
    Config,
    Adapter,
    PredictionInverter,
    MLOutputApplier,
)
from runtime.transformers.core import StepTransformer
from machine_learning_mocks import get_mock_predictor


TEMP = "air_temperature"
SPHUM = "specific_humidity"
Q1 = "Q1"
Q2 = "Q2"
DQ1 = "dQ1"
DQ2 = "dQ2"
NUDGING = "tendency_of_air_temperature_due_to_nudging"
PRECIP_PRED = "implied_surface_precipitation_rate"
PRECIP = "surface_precipitation_rate"
DOWN_SW_PREDICTION_NAME = "total_sky_downward_shortwave_flux_at_surface"
NZ = 63


class MockDerivedState:
    def __init__(self, ds: xr.Dataset):
        self._mapper = ds

    def __getitem__(self, key: str) -> xr.DataArray:
        return vcm.DerivedMapping(self._mapper)[key]

    def __setitem__(self, key: str, value: xr.DataArray):
        self._mapper[key] = value

    def update(self, items):
        for key, value in items.items():
            self._mapper[key] = value


def regression_state(state, regtest):
    for v in sorted(state):
        print(v, joblib.hash(state[v].values), file=regtest)


def test_adapter_regression(state, regtest, tmpdir_factory):
    model_path = str(tmpdir_factory.mktemp("model"))
    mock = get_mock_predictor(dQ1_tendency=1 / 86400)
    fv3fit.dump(mock, model_path)

    adapted_model = Adapter(
        Config(
            [model_path],
            output_targets={
                DQ1: MLOutputApplier(TEMP, "tendency"),
                DQ2: MLOutputApplier(SPHUM, "tendency"),
            },
            limit_negative_humidity=True,
        ),
        900,
    )
    transform = StepTransformer(
        adapted_model,
        MockDerivedState(state),
        "machine_learning",
        diagnostic_variables={
            "tendency_of_specific_humidity_due_to_machine_learning",
            "tendency_of_air_temperature_due_to_machine_learning",
            "tendency_of_internal_energy_due_to_machine_learning",
        },
        timestep=900,
    )

    def add_one_to_temperature():
        state["air_temperature"] += 1
        return {"some_diag": state["specific_humidity"]}

    out = transform(add_one_to_temperature)()

    # ensure tendency of internal energy is non-zero somewhere (GH#1433)
    max = abs(out["tendency_of_internal_energy_due_to_machine_learning"]).max()
    assert max.values.item() > 1e-6

    # sort to make the check deterministic
    regression_state(out, regtest)


def test_prediction_inverter():
    inverter = PredictionInverter(
        {
            Q1: MLOutputApplier(TEMP, "tendency"),
            Q2: MLOutputApplier(SPHUM, "tendency"),
            NUDGING: MLOutputApplier(TEMP, "tendency"),
            PRECIP_PRED: MLOutputApplier(PRECIP, "state"),
        }
    )
    prediction = {
        Q1: xr.DataArray(np.full(NZ, 1 / 86400), dims=("z")),
        NUDGING: xr.DataArray(np.full(NZ, 1 / 86400), dims=("z")),
        Q2: xr.DataArray(np.full(NZ, 1 / 86400 / 1000), dims=("z")),
        PRECIP_PRED: xr.DataArray(np.full(NZ, 1 / 86400), dims=("z")),
    }
    inverted_prediction = inverter.invert(prediction)
    assert TEMP in inverted_prediction
    assert PRECIP in inverted_prediction
    xr.testing.assert_allclose(
        inverted_prediction[TEMP], xr.DataArray(np.full(NZ, 2 / 86400), dims=("z"))
    )


def test_prediction_inverter_raises_multiple_state_update_error():
    with pytest.raises(ValueError):
        PredictionInverter(
            {Q1: MLOutputApplier(TEMP, "state"), Q2: MLOutputApplier(TEMP, "state")}
        )


def test_prediction_inverter_raises_state_and_tendency_error():
    with pytest.raises(ValueError):
        PredictionInverter(
            {Q1: MLOutputApplier(TEMP, "tendency"), Q2: MLOutputApplier(TEMP, "state")}
        )


def test_multimodel_adapter_integration(state, tmpdir_factory):
    model_path1 = str(tmpdir_factory.mktemp("model1"))
    model_path2 = str(tmpdir_factory.mktemp("model2"))
    mock1 = get_mock_predictor(dQ1_tendency=1 / 86400)
    mock2 = get_mock_predictor(model_predictands="rad_fluxes")
    fv3fit.dump(mock1, model_path1)
    fv3fit.dump(mock2, model_path2)

    adapted_model = Adapter(
        Config(
            [model_path1, model_path2],
            output_targets={
                DQ1: MLOutputApplier(TEMP, "tendency"),
                DOWN_SW_PREDICTION_NAME: MLOutputApplier(PRECIP, "state"),
            },
            limit_negative_humidity=False,
        ),
        900,
    )
    transform = StepTransformer(
        adapted_model, MockDerivedState(state), "machine_learning", timestep=900,
    )

    def add_one_to_temperature():
        state["air_temperature"] += 1
        return {"some_diag": state["specific_humidity"]}

    assert "surface_precipitation_rate" not in state
    transform(add_one_to_temperature)()
    assert "surface_precipitation_rate" in state
