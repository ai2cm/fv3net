import joblib
import fv3fit
from runtime.transformers.fv3fit import Config, Adapter
from runtime.transformers.core import StepTransformer
from machine_learning_mocks import get_mock_predictor


def regression_state(state, regtest):
    for v in sorted(state):
        print(v, joblib.hash(state[v].values), file=regtest)


def test_adapter_regression(state, regtest, tmpdir_factory):
    model_path = str(tmpdir_factory.mktemp("model"))
    mock = get_mock_predictor()
    fv3fit.dump(mock, model_path)

    adapted_model = Adapter(Config(model_path, {"specific_humidity": "dQ2"}), 900)
    transform = StepTransformer(
        adapted_model,
        state,
        "machine_learning",
        diagnostic_variables={"tendency_of_specific_humidity_due_to_machine_learning"},
        timestep=900,
    )

    def add_one_to_temperature():
        state["air_temperature"] += 1
        return {"some_diag": state["specific_humidity"]}

    out = transform(add_one_to_temperature)()

    # sort to make the check deterministic
    regression_state(out, regtest)
