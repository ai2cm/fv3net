import argparse
import os
import yaml
import logging

import fv3config
import fv3kube
import vcm

logger = logging.getLogger(__name__)


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "initial_condition_url",
        type=str,
        help="Remote url to directory holding timesteps with model initial conditions.",
    )
    parser.add_argument(
        "ic_timestep",
        type=str,
        help="Time step to grab from the initial conditions url.",
    )
    parser.add_argument(
        "output_url",
        type=str,
        help="Remote storage location for prognostic run output.",
    )
    parser.add_argument(
        "--model_url", type=str, default=None, help="Remote url to a trained ML model.",
    )
    parser.add_argument(
        "--nudge-to-observations", action="store_true", help="Nudge to observations",
    )
    parser.add_argument(
        "--prog_config_yml",
        type=str,
        default="prognostic_config.yml",
        help="Path to a config update YAML file specifying the changes from the base"
        "fv3config (e.g. diag_table, runtime, ...) for the prognostic run.",
    )
    parser.add_argument(
        "--diagnostic_ml",
        action="store_true",
        help="Compute and save ML predictions but do not apply them to model state.",
    )
    return parser


def insert_ml_settings(model_config, model_url, diagnostic_ml):

    # Add scikit learn ML model config section
    scikit_learn_config = model_config.get("scikit_learn", {})
    scikit_learn_config["zarr_output"] = "diags.zarr"
    model_config.update(scikit_learn=scikit_learn_config)

    if model_url:
        # insert the model asset
        model_type = scikit_learn_config.get("model_type", "scikit_learn")
        if model_type == "scikit_learn":
            _update_sklearn_config(model_config, model_url)
        elif model_type == "keras":
            _update_keras_config(model_config, model_url)
        else:
            raise ValueError(
                "Available model types are 'scikit_learn' and 'keras'; received type:"
                f" {model_type}."
            )
        model_config["scikit_learn"].update(diagnostic_ml=diagnostic_ml)


def _update_sklearn_config(
    model_config, model_url, sklearn_filename="sklearn_model.pkl"
):
    model_asset = fv3config.get_asset_dict(
        model_url, sklearn_filename, target_name=sklearn_filename
    )
    model_config.setdefault("patch_files", []).append(model_asset)
    model_config["scikit_learn"].update(model=sklearn_filename)


def _update_keras_config(model_config, model_url, keras_dirname="model_data"):
    model_asset_list = fv3config.asset_list_from_path(
        os.path.join(args.model_url, keras_dirname), target_location=keras_dirname
    )
    model_config.setdefault("patch_files", []).extend(model_asset_list)
    model_config["scikit_learn"].update(model=keras_dirname)


def insert_default_diagnostics(model_config):
    """ Inserts default diagnostics save configuration into base config,
    which is overwritten by user-provided config if diagnostics entry is present.
    Defaults to of saving original 2d diagnostics on 15 min frequency.
    If variables in this list does not exist, e.g. *_diagnostic vars only exist
    if --diagnostic_ml flag is set, they are skipped.

    Args:
        model_config: Prognostic run configuration dict
    """
    model_config["diagnostics"] = [
        {
            "name": "diags.zarr",
            "variables": [
                "net_moistening",
                "net_moistening_diagnostic",
                "net_heating",
                "net_heating_diagnostic",
                "water_vapor_path",
                "physics_precip",
            ],
            "times": {"kind": "interval", "frequency": 900},
        }
    ]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = _create_arg_parser()
    args = parser.parse_args()

    # Get model config with prognostic run updates
    with open(args.prog_config_yml, "r") as f:
        user_config = yaml.safe_load(f)

    # It should be possible to implement all configurations as overlays
    # so this could be done as one vcm.update_nested_dict call
    # updated_nested_dict just needs to know how to merge patch_files fields
    config = vcm.update_nested_dict(
        fv3kube.get_base_fv3config(user_config.get("base_version")),
        fv3kube.c48_initial_conditions_overlay(
            args.initial_condition_url, args.ic_timestep
        ),
        {"diag_table": "/fv3net/workflows/prognostic_c48_run/diag_table_prognostic"},
    )

    insert_default_diagnostics(config)

    insert_ml_settings(user_config, args.model_url, args.diagnostic_ml)

    model_config = vcm.update_nested_dict(
        config,
        # User settings override previous ones
        user_config,
    )

    if args.nudge_to_observations:
        model_config = fv3kube.enable_nudge_to_observations(model_config)

    print(model_config)
