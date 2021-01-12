import argparse
import yaml
import logging
from typing import Sequence, Dict, List

import fv3config
import fv3kube

import vcm

from runtime import default_diagnostics

logger = logging.getLogger(__name__)

PROGNOSTIC_DIAG_TABLE = "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "user_config",
        type=str,
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        help="YYYYMMDD.HHMMSS timestamp to grab from the initial conditions url.",
    )
    parser.add_argument(
        "--model_url",
        type=str,
        default=None,
        action="append",
        help=(
            "Remote url to a trained ML model. If a model is omitted (and not "
            "specified in `user_config`'s `scikit-learn` `model` field either), then "
            "no ML updating will be done. Also, if an ML model is provided, no "
            "nudging will be done. Can be provided multiple times, "
            "ex. --model_url model1 --model_url model2. If multiple urls are given, "
            "they will be combined into a single model at runtime, providing the "
            "outputs are nonoverlapping."
        ),
    )
    parser.add_argument(
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
    )
    parser.add_argument(
        "--output-frequency",
        type=int,
        default=15,
        help=(
            "Output frequency (in minutes) of ML/nudging diagnostics. If omitted, "
            "output will be written every 15 minutes from the initial time."
        ),
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def ml_overlay(model_urls: List[str], diagnostic_ml: bool) -> dict:
    if len(model_urls) > 0:
        overlay = {"scikit_learn": {"model": model_urls}}
        overlay["scikit_learn"].update({"diagnostic_ml": diagnostic_ml})  # type: ignore
    else:
        overlay = {}
    return overlay


def nudging_overlay(nudging_config, initial_condition_url):
    if "timescale_hours" in nudging_config:
        nudging_config.update({"restarts_path": initial_condition_url})
        overlay = {"nudging": nudging_config}
    else:
        overlay = {}
    return overlay


def diagnostics_overlay(
    config: dict, model_urls: List[str], nudge_to_obs: bool, frequency_minutes: int
):

    diagnostic_files = []  # type: List[Dict]

    if ("scikit_learn" in config) or len(model_urls) > 0:
        diagnostic_files.append(default_diagnostics.ml_diagnostics.to_dict())
    elif "nudging" in config or nudge_to_obs:
        diagnostic_files.append(default_diagnostics.state_after_timestep.to_dict())
        diagnostic_files.append(default_diagnostics.physics_tendencies.to_dict())
        if "nudging" in config:
            diagnostic_files.append(_nudging_tendencies(config))
            diagnostic_files.append(
                default_diagnostics.nudging_diagnostics_2d.to_dict()
            )
            diagnostic_files.append(_reference_state(config))
    else:
        diagnostic_files.append(default_diagnostics.baseline_diagnostics.to_dict())

    diagnostic_files = _update_times(diagnostic_files, frequency_minutes)

    return {"diagnostics": diagnostic_files, "diag_table": PROGNOSTIC_DIAG_TABLE}


def _nudging_tendencies(config):

    nudging_tendencies = default_diagnostics.nudging_tendencies.to_dict()
    nudging_variables = list(config["nudging"]["timescale_hours"])
    nudging_tendencies["variables"].extend(
        [f"{var}_tendency_due_to_nudging" for var in nudging_variables]
    )
    return nudging_tendencies


def _reference_state(config):
    reference_states = default_diagnostics.reference_state.to_dict()
    nudging_variables = list(config["nudging"]["timescale_hours"])
    reference_states["variables"].extend(
        [f"{var}_reference" for var in nudging_variables]
    )
    return reference_states


def _update_times(diagnostic_files: List[Dict], frequency_minutes: int) -> List[Dict]:
    for diagnostic in diagnostic_files:
        diagnostic.update(
            {"times": {"kind": "interval", "frequency": 60 * frequency_minutes}}
        )
    return diagnostic_files


def step_tendency_overlay(
    config,
    default_step_tendency_variables=(
        "specific_humidity",
        "air_temperature",
        "eastward_wind",
        "northward_wind",
    ),
    default_step_storage_variables=("specific_humidity", "total_water"),
):
    step_tendency_overlay = {}
    step_tendency_overlay["step_tendency_variables"] = config.get(
        "step_tendency_variables", list(default_step_tendency_variables)
    )
    step_tendency_overlay["step_storage_variables"] = config.get(
        "step_storage_variables", list(default_step_storage_variables)
    )
    return step_tendency_overlay


def prepare_config(args):
    # Get model config with prognostic run updates
    with open(args.user_config, "r") as f:
        user_config = yaml.safe_load(f)

    model_urls = args.model_url if args.model_url else []
    nudging_config = user_config.get("nudging", {})

    # To simplify the configuration flow, updates should be implemented as
    # overlays (i.e. diffs) requiring only a small number of inputs. In
    # particular, overlays should not require access to the full configuration
    # dictionary.
    overlays = [
        fv3kube.get_base_fv3config(user_config.get("base_version")),
        fv3kube.c48_initial_conditions_overlay(
            args.initial_condition_url, args.ic_timestep
        ),
        diagnostics_overlay(
            user_config, model_urls, args.nudge_to_observations, args.output_frequency,
        ),
        step_tendency_overlay(user_config),
        ml_overlay(model_urls, args.diagnostic_ml),
        nudging_overlay(nudging_config, args.initial_condition_url),
        user_config,
    ]

    if args.nudge_to_observations:
        # get timing information
        duration = fv3config.get_run_duration(user_config)
        current_date = vcm.parse_current_date_from_str(args.ic_timestep)
        overlays.append(
            fv3kube.enable_nudge_to_observations(
                duration,
                current_date,
                nudge_url="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
            )
        )

    config = fv3kube.merge_fv3config_overlays(*overlays)
    print(yaml.dump(config))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()
    prepare_config(args)
