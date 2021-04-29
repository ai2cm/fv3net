import prepare_config
import pytest
import dacite
import dataclasses
from runtime.config import UserConfig

MODEL_URL = "gs://ml-model"
IC_URL = "gs://ic-bucket"
IC_TIMESTAMP = "20160805.000000"
ML_CONFIG_UPDATE = "prognostic_config.yml"
NUDGE_TO_FINE_CONFIG_UPDATE = "nudge_to_fine_config.yml"
OTHER_FLAGS = ["--nudge-to-observations"]


def get_ml_args():
    return [
        ML_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
        "--model_url",
        MODEL_URL,
    ] + OTHER_FLAGS


def get_nudge_to_fine_args():
    return [
        NUDGE_TO_FINE_CONFIG_UPDATE,
        IC_URL,
        IC_TIMESTAMP,
    ]


def test_prepare_ml_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_ml_args())
    with regtest:
        prepare_config.prepare_config(args)


def test_prepare_nudging_config_regression(regtest):
    parser = prepare_config._create_arg_parser()
    args = parser.parse_args(get_nudge_to_fine_args())
    with regtest:
        prepare_config.prepare_config(args)


@pytest.mark.parametrize(
    ["frequency_minutes", "expected"],
    [
        pytest.param(
            120,
            {
                "kind": "interval",
                "frequency": 7200,
                "times": None,
                "includes_lower": False,
            },
            id="2-hourly",
        ),
        pytest.param(
            15,
            {
                "kind": "interval",
                "frequency": 900,
                "times": None,
                "includes_lower": False,
            },
            id="default_15-minute",
        ),
    ],
)
def test_diagnostics_overlay_times(frequency_minutes, expected):
    class Args:
        model_url = []
        diagnostic_ml = True

    config = prepare_config.user_config_from_dict_and_args(
        {"base_version": "v0.5"}, Args
    )

    diags_overlay_times = prepare_config.diagnostics_overlay(
        config, [], None, frequency_minutes
    )["diagnostics"][0]["times"]
    assert diags_overlay_times == expected


def test_get_user_config_is_valid():
    class Args:
        model_url = []
        diagnostic_ml = True
        initial_condition_url = "gs://some-url"
        ic_timestep = "20160801.000000"
        nudge_to_observations = False
        output_frequency = 900

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
