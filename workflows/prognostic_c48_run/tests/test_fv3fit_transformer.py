import joblib
import fv3fit
import vcm
import numpy as np
import xarray as xr
import pytest
from runtime.transformers.fv3fit import Config, Adapter, _limit_ds
from runtime.transformers.core import StepTransformer
from machine_learning_mocks import get_mock_predictor


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
            tendency_predictions={"dQ1": "air_temperature", "dQ2": "specific_humidity"},
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

    temperature_before = state["air_temperature"].copy(deep=True)
    out = transform(add_one_to_temperature)()
    temperature_after = state["air_temperature"].copy(deep=True)

    # ensure tendency of internal energy is non-zero somewhere (GH#1433)
    max = abs(out["tendency_of_internal_energy_due_to_machine_learning"]).max()
    assert max.values.item() > 1e-6

    np.testing.assert_allclose(
        temperature_after, temperature_before + 900 / 86400, rtol=1e-5
    )

    # sort to make the check deterministic
    regression_state(out, regtest)


def test_config_raises_multiple_state_update_error():
    with pytest.raises(ValueError):
        Config(["gs://bucket"], state_predictions={"foo": "bar", "baz": "bar"})


def test_config_raises_state_and_tendency_error():
    with pytest.raises(ValueError):
        Config(
            ["gs://bucket"],
            tendency_predictions={"foo": "bar"},
            state_predictions={"baz": "bar"},
        )


def test_multimodel_adapter_integration(state, tmpdir_factory):
    DOWN_SW_PREDICTION_NAME = "total_sky_downward_shortwave_flux_at_surface"
    model_path1 = str(tmpdir_factory.mktemp("model1"))
    model_path2 = str(tmpdir_factory.mktemp("model2"))
    mock1 = get_mock_predictor(dQ1_tendency=1 / 86400)
    mock2 = get_mock_predictor(model_predictands="rad_fluxes")
    fv3fit.dump(mock1, model_path1)
    fv3fit.dump(mock2, model_path2)

    adapted_model = Adapter(
        Config(
            [model_path1, model_path2],
            tendency_predictions={"dQ1": "air_temperature", "dQ2": "air_temperature"},
            state_predictions={DOWN_SW_PREDICTION_NAME: "surface_precipitation_rate"},
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


def test__limit_ds_missing():
    with pytest.raises(KeyError):
        _limit_ds(xr.Dataset({"a": xr.DataArray(1)}), {"b": 2}, {})


def test__limit_ds():
    input_ds = xr.Dataset({"a": xr.DataArray([1, 1]), "b": xr.DataArray([-3, 1])})
    expected_limited_ds = xr.Dataset(
        {"a": xr.DataArray([0.5, 0.5]), "b": xr.DataArray([-2, 0])}
    )
    _limit_ds(input_ds, {"b": -2}, {"a": 0.5, "b": 0})
    xr.testing.assert_identical(input_ds, expected_limited_ds)
