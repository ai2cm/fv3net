from datetime import datetime
import argparse
import os
import sys
import yaml

import fv3kube
import vcm

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)
import runfile  # noqa


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudging run"
    )
    parser.add_argument("nudging_config", type=str, help="base yaml file to configure")
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
    with open(args.nudging_config, "r") as f:
        config = yaml.safe_load(f)

    reference_dir = config["nudging"]["restarts_path"]
    time = datetime(*config["namelist"]["coupler_nml"]["current_date"])
    label = vcm.encode_time(time)
    config = fv3kube.get_full_config(config, reference_dir, label)

    if args.timescale_hours is not None:
        for varname in config["nudging"]["timescale_hours"]:
            config["nudging"]["timescale_hours"][varname] = args.timescale_hours
            
    with open(args.outfile, "w") as f:
        f.write(yaml.dump(config))
