import yaml
import argparse
import os

import fv3kube
import fv3config


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudging run"
    )
    parser.add_argument("config", type=str, help="base yaml file to configure")
    parser.add_argument(
        "restart_dir", type=str, help="location of initial conditions folders"
    )
    parser.add_argument(
        "ic_timestep", type=str, help="YYYYMMDD.HHMMSS coded initial timestep"
    )
    parser.add_argument("rundir", type=str, help="Location to place run directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)
    
    full_config = fv3kube.get_full_config(base_cfg, args.restart_dir, args.ic_timestep)
    fv3config.write_run_directory(full_config, args.rundir)
