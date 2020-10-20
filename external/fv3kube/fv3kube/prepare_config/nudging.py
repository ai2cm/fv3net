from datetime import datetime
import yaml
import argparse

import fv3kube

DEFAULT_CHUNKS = {
    "before_dynamics.zarr": {"time": 1},
    "after_dynamics.zarr": {"time": 1},
    "after_physics.zarr": {"time": 1},
    "after_nudging.zarr": {"time": 1},
    "reference.zarr": {"time": 1},
    "nudging_tendencies.zarr": {"time": 1},
    "atmos_output.zarr": {"time": 24},
    "physics_output.zarr": {"time": 24},
    "physics_tendency_components.zarr": {"time": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudging run"
    )
    parser.add_argument("config", type=str, help="base yaml file to configure")
    parser.add_argument(
        "--timesteps",
        type=str,
        help="path to yaml-encoded list of YYYYMMDD.HHMMSS timesteps",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.timesteps:
        with open(args.timesteps) as f:
            timesteps = yaml.safe_load(f)

        if timesteps is not None:
            config["nudging"]["output_times"] = timesteps

    reference_dir = config["nudging"]["restarts_path"]
    time = datetime(*config["namelist"]["coupler_nml"]["current_date"])
    label = fv3kube.time.encode_time(time)
    config = fv3kube.get_full_config(config, reference_dir, label)
    print(yaml.dump(config))


def default_chunks():
    print(yaml.safe_dump(DEFAULT_CHUNKS))
