from datetime import datetime
import os
import sys
import argparse
from typing import List
import fv3config
import yaml

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)
import runfile  # noqa

FV_CORE_ASSET = fv3config.get_asset_dict(
    "gs://vcm-fv3config/data/initial_conditions/fv_core_79_levels/v1.0/",
    "fv_core.res.nc",
    target_location="INPUT",
)

RESTART_CATEGORIES = ["fv_core.res", "sfc_data", "fv_tracer.res", "fv_srf_wnd.res"]
TILE_COORDS_FILENAMES = range(1, 7)  # tile numbering in model output filenames


def get_initial_condition_assets(input_url: str, label: str) -> List[dict]:
    """Get list of assets representing initial conditions for this label"""
    # If you re-use this somewhere else or have time to refactor it, please refactor
    # it to use fv3net/pipelines/kube_jobs/utils.py#L42
    # (update_tiled_asset_names)
    initial_condition_assets = [
        fv3config.get_asset_dict(
            os.path.join(input_url, label),
            f"{label}.{category}.tile{tile}.nc",
            target_location="INPUT",
            target_name=f"{category}.tile{tile}.nc",
        )
        for category in RESTART_CATEGORIES
        for tile in TILE_COORDS_FILENAMES
    ]
    return initial_condition_assets


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudging run"
    )
    parser.add_argument("basefile", type=str, help="base yaml file to configure")
    parser.add_argument("outfile", type=str, help="location to write yaml file")
    parser.add_argument(
        "--timescale-hours",
        type=float,
        default=None,
        help="timescale for nudging in hours",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.basefile, "r") as f:
        config = yaml.safe_load(f)
    reference_dir = config["nudging"]["restarts_path"]
    time = datetime(*config["namelist"]["coupler_nml"]["current_date"])
    label = runfile.time_to_label(time)
    config["initial_conditions"] = get_initial_condition_assets(reference_dir, label)
    config["initial_conditions"].append(FV_CORE_ASSET)
    if args.timescale_hours is not None:
        for varname in config["nudging"]["timescale_hours"]:
            config["nudging"]["timescale_hours"][varname] = args.timescale_hours
    with open(args.outfile, "w") as f:
        f.write(yaml.dump(config))
