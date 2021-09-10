from workflows.prognostic_c48_run.runtime.segmented_run.prepare_config import (
    to_fv3config,
)
from runtime.segmented_run import prepare_config
import dacite
import dataclasses
from runtime.config import UserConfig
import pytest


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(
            ["examples/prognostic_config.yml", "--model_url", "gs://ml-model"], id="ml"
        ),
        pytest.param(["examples/nudge_to_fine_config.yml"], id="n2f"),
        pytest.param(["examples/nudge_to_obs_config.yml"], id="n2o"),
        pytest.param(["examples/emulator.yml"], id="emulator"),
    ],
)
def test_prepare_ml_config_regression(regtest, argv):
    IC_URL = "gs://ic-bucket"
    IC_TIMESTAMP = "20160805.000000"

    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(argv + [IC_URL, IC_TIMESTAMP])
    with regtest:
        prepare_config.prepare_config(args)


def test_get_user_config_is_valid():

    dict_ = {
        "base_version": "v0.5",
        "diagnostics": [
            {
                "name": "state_after_timestep.zarr",
                "times": {"frequency": 5400, "kind": "interval", "times": None},
                "variables": ["x_wind", "y_wind"],
            }
        ],
    }

    config = prepare_config.user_config_from_dict_and_args(
        dict_, model_url=[], diagnostic_ml=True, nudging_url="gs://some-url",
    )
    # validate using dacite.from_dict
    dacite.from_dict(UserConfig, dataclasses.asdict(config))


def test_to_fv3config_initial_conditions():
    my_ic = "my_ic"
    final = to_fv3config(
        {"initial_conditions": my_ic, "base_version": "v0.5"},
        initial_condition=None,
        model_url=[],
        diagnostic_ml=True,
        nudging_url="gs://some-url",
    )

    assert final["initial_conditions"] == my_ic
