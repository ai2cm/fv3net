import argparse
import os
import yaml
import logging

import fv3config
import fv3kube

import vcm

logger = logging.getLogger(__name__)


DEFAULT_ML_DIAGNOSTICS = {
    "name": "ML_diags.zarr",
    "output_variables": [
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
    "times": {"kind": "interval", "frequency": 900},
}


DEFAULT_NUDGING_DIAGNOSTICS = {
    "name": "nudging_diags.zarr",
    "output_variables": [
        "net_moistening_due_to_nudging",
        "net_heating_due_to_nudging",
        "net_mass_tendency_due_to_nudging",
        "column_integrated_u-wind_tendency_due_to_nudging",
        "column_integrated_v-wind_tendency_due_to_nudging",
        "water_vapor_path",
        "physics_precip",
        "tendency_of_air_temperature_due_to_fv3_physics",
        "tendency_of_specific_humidity_due_to_fv3_physics",
    ],
    "times": {"kind": "interval", "frequency": 900},
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
    parser.add_argument(
        "--output-timestamps",
        type=str,
        default=None,
        help=(
            "path to yaml-encoded list of YYYYMMDD.HHMMSS timestamps, which define "
            "a subset of run's timestamps that will be written to disk. If ommitted "
            "timestamps will be written every 15 minutes from the initial time."
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


def diagnostics_overlay(config, model_url, timestamps):
    diagnostic_files = []
    if timestamps:
        for diagnostics in [DEFAULT_ML_DIAGNOSTICS, DEFAULT_NUDGING_DIAGNOSTICS]:
            diagnostics["times"]["kind"] = "selected"
            diagnostics["times"]["times"] = timestamps
    if ("scikit_learn" in config) or model_url:
        diagnostic_files.append(DEFAULT_ML_DIAGNOSTICS)
    if "nudging" in config:
        nudging_variables = list(config["nudging"]["timescale_hours"])
        nudging_diagnostics = DEFAULT_NUDGING_DIAGNOSTICS
        nudging_diagnostics["output_variables"].extend(
            [f"{var}_tendency_due_to_nudging" for var in nudging_variables]
        )
        diagnostic_files.append(nudging_diagnostics)
    return {
        "diagnostics": diagnostic_files,
        "diag_table": "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic",
    }


def tendency_overlay(config):
    tendency_overlay = {}
    if "tendency_variables" in config:
        tendency_overlay.update({"tendency_variables": config["tendency_variables"]})
    if "storage_variables" in config:
        tendency_overlay.update({"storage_variables": config["storage_variables"]})
    return tendency_overlay


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
        diagnostics_overlay(user_config, args.model_url, timestamps),
        tendency_overlay(user_config),
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
