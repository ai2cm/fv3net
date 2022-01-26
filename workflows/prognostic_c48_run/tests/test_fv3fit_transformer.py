import joblib
import fv3fit
import vcm
import xarray as xr
from runtime.transformers.fv3fit import Config, Adapter
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
            model_path,
            {"air_temperature": "dQ1", "specific_humidity": "dQ2"},
            bias_correction_factor={"dQ1": -1},
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
