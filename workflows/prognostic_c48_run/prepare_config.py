import argparse
import os
import yaml
import logging

import fv3config
import fv3kube

import vcm

logger = logging.getLogger(__name__)


TIMES = {"kind": "interval", "frequency": 900}
ML_DIAGNOSTICS = {
    "name": "diags.zarr",
    "variables": [
        "net_moistening",
        "net_moistening_diagnostic",
        "net_heating",
        "net_heating_diagnostic",
        "water_vapor_path",
        "physics_precip",
        "column_integrated_dQu",
        "column_integrated_dQu_diagnostic",
        "column_integrated_dQv",
        "column_integrated_dQv_diagnostic",
    ],
}
NUDGING_DIAGNOSTICS_2D = {
    "name": "diags.zarr",
    "variables": [
        "net_moistening_due_to_nudging",
        "net_heating_due_to_nudging",
        "net_mass_tendency_due_to_nudging",
        "column_integrated_eastward_wind_tendency_due_to_nudging",
        "column_integrated_northward_wind_tendency_due_to_nudging",
        "water_vapor_path",
        "physics_precip",
    ],
}
NUDGING_TENDENCIES = {"name": "nudging_tendencies.zarr", "variables": []}
BASELINE_DIAGNOSTICS = {
    "name": "diags.zarr",
    "variables": ["water_vapor_path", "physics_precip"],
}


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
        "--model_url", type=str, default=None, help="Remote url to a trained ML model.",
    )
    parser.add_argument(
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--output-timestamps",
        type=str,
        default=None,
        help=(
            "Path to yaml-encoded list of YYYYMMDD.HHMMSS timestamps, which define "
            "a subset of run's timestamps that will be written to disk. Mutually "
            "exclusive with `output-frequency`. If both are omitted, timestamps will "
            "be written every 15 minutes from the initial time."
        ),
    )
    group.add_argument(
        "--output-frequency",
        type=int,
        default=None,
        help=(
            "Output frequency (in minutes) of ML/nudging diagnostics. Mutually "
            "exclusive with `output-timestamps`. If both are omitted, timestamps "
            "will be written every 15 minutes from the initial time."
        ),
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def ml_overlay(model_type, model_url, diagnostic_ml):
    if model_url:
        if model_type == "scikit_learn":
            overlay = sklearn_overlay(model_url)
        elif model_type == "keras":
            overlay = keras_overlay(model_url)
        else:
            raise ValueError(
                "Available model types are 'scikit_learn' and 'keras'; received type:"
                f" {model_type}."
            )
        overlay["scikit_learn"].update({"diagnostic_ml": diagnostic_ml}),
    else:
        overlay = {}
    return overlay


def sklearn_overlay(model_url):
    return {"scikit_learn": {"model": model_url}}


def keras_overlay(model_url, keras_dirname="model_data"):
    model_asset_list = fv3config.asset_list_from_path(
        os.path.join(model_url, keras_dirname), target_location=keras_dirname
    )
    return {"patch_files": model_asset_list, "scikit_learn": {"model": keras_dirname}}


def nudging_overlay(nudging_config, initial_condition_url):
    if "timescale_hours" in nudging_config:
        nudging_config.update({"restarts_path": initial_condition_url})
        overlay = {"nudging": nudging_config}
    else:
        overlay = {}
    return overlay


def diagnostics_overlay(config, model_url, timestamps, frequency_minutes):

    diagnostic_files = []

    if ("scikit_learn" in config) or model_url:
        diagnostic_files.append(ML_DIAGNOSTICS)
    elif "nudging" in config:
        nudging_tendencies = NUDGING_TENDENCIES
        nudging_variables = list(config["nudging"]["timescale_hours"])
        nudging_tendencies["variables"].extend(
            [f"{var}_tendency_due_to_nudging" for var in nudging_variables]
        )
        diagnostic_files.append(nudging_tendencies)
        diagnostic_files.append(NUDGING_DIAGNOSTICS_2D)
    else:
        diagnostic_files.append(BASELINE_DIAGNOSTICS)

    for diagnostic in diagnostic_files:
        if timestamps:
            diagnostic.update({"times": {"kind": "selected", "times": timestamps}})
        elif frequency_minutes:
            diagnostic.update(
                {"times": {"kind": "interval", "frequency": 60 * frequency_minutes}}
            )
        else:
            diagnostic.update({"times": TIMES})

    return {
        "diagnostics": diagnostic_files,
        "diag_table": "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic",
    }


def step_tendency_overlay(
    config,
    default_step_tendency_variables=("specific_humidity", "air_temperature"),
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

    model_type = user_config.get("scikit_learn", {}).get("model_type", "scikit_learn")
    nudging_config = user_config.get("nudging", {})

    if args.output_timestamps:
        with open(args.output_timestamps) as f:
            timestamps = yaml.safe_load(f)
    else:
        timestamps = None

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
            user_config, args.model_url, timestamps, args.output_frequency
        ),
        step_tendency_overlay(user_config),
        ml_overlay(model_type, args.model_url, args.diagnostic_ml),
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
