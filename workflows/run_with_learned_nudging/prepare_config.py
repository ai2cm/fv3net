import argparse
import logging
import os
import fsspec
import yaml
import vcm
from fv3net.pipelines.kube_jobs import (
    get_base_fv3config,
    update_nested_dict,
    update_config_for_nudging,
)

logger = logging.getLogger("run_jobs")

BUCKET = "gs://vcm-ml-data/2020-01-29-baseline-FV3GFS-runs"
CATEGORY = "diagnostic_zarr/atmos_monthly"

NUDGING_TENDENCY_URL = {
    "T": os.path.join(BUCKET, "nudged-T-2015-C48-npz63-fv3atm", CATEGORY),
    "T_ps": os.path.join(BUCKET, "nudged-T-ps-2015-C48-npz63-fv3atm", CATEGORY),
    "T_ps_u_v": os.path.join(BUCKET, "nudged-T-ps-u-v-2015-C48-npz63-fv3atm", CATEGORY),
}

VARIABLES_TO_NUDGE = {
    "T": ["air_temperature"],
    "T_ps": ["air_temperature", "pressure_thickness_of_atmospheric_layer"],
    "T_ps_u_v": [
        "air_temperature",
        "pressure_thickness_of_atmospheric_layer",
        "eastward_wind_after_physics",
        "northward_wind_after_physics",
    ],
}

EXPERIMENT_NAME = {
    "T": "nudge-mean-t",
    "T_ps": "nudge-mean-t-ps",
    "T_ps_u_v": "nudge-mean-t-ps-u-v",
}


def prepare_config(template, base_config, nudge_label, config_url):
    """Get config objects for current job and upload as necessary"""
    config = update_nested_dict(base_config, template)
    config["runtime"]["nudging_zarr_url"] = NUDGING_TENDENCY_URL[nudge_label]
    config["runtime"]["variables_to_nudge"] = VARIABLES_TO_NUDGE[nudge_label]
    config["experiment_name"] = EXPERIMENT_NAME[nudge_label]
    if config["namelist"]["fv_core_nml"].get("nudge", False):
        config = update_config_for_nudging(config, config_url)
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_template",
        type=str,
        help="Path to fv3config yaml template.",
    )
    parser.add_argument(
        "nudge_label",
        type=str,
        help="Label for variables to nudge. Must be one of 'T', 'T_ps' or 'T_ps_u_v'.",
    )
    parser.add_argument(
        "config_url",
        type=str,
        help="Local or remote location where config files will be saved for later use.",
    )
    parser.add_argument(
        "--config-version",
        type=str,
        required=False,
        default="v0.3",
        help="Default fv3config.yml version to use as the base configuration. "
        "This should be consistent with the fv3gfs-python version in the specified "
        "docker image.",
    )
    args = parser.parse_args()
    base_config = get_base_fv3config(args.config_version)
    with open(args.config_template) as f:
        template = yaml.load(f, Loader=yaml.FullLoader)
    config = prepare_config(template, base_config, args.nudge_label, args.config_url)
    fs = vcm.cloud.get_fs(args.config_url)
    fs.mkdirs(args.config_url, exist_ok=True)
    with fsspec.open(os.path.join(args.config_url, "fv3config.yml"), "w") as f:
        yaml.safe_dump(config, f)
