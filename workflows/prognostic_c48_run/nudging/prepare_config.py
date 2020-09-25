from datetime import datetime
import os
import sys
import yaml
import argparse

import fv3kube
import vcm


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


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)


if __name__ == "__main__":

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
    label = vcm.encode_time(time)
    config = fv3kube.get_full_config(config, reference_dir, label)
    print(yaml.dump(config))
