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
    class Args:
        model_url = []
        diagnostic_ml = True
        initial_condition_url = "gs://some-url"
        ic_timestep = "20160801.000000"
        nudge_to_observations = False

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

    config = prepare_config.user_config_from_dict_and_args(dict_, Args)
    # validate using dacite.from_dict
    dacite.from_dict(UserConfig, dataclasses.asdict(config))
