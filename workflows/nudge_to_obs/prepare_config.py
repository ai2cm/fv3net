from datetime import datetime
import os
import sys
import yaml
import argparse

import fv3kube
import vcm


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudge-to-obs run"
    )
    parser.add_argument("config", type=str, help="base yaml file to configure")
    parser.add_argument(
        "rundir", type=str, help="path to rundir, required for writing nudging filelist"
    )

    return parser.parse_args()


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)


if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r") as f:
        config_update = yaml.safe_load(f)

    base_config = fv3kube.get_base_fv3config(config_update["base_version"])
    config = vcm.update_nested_dict(base_config, config_update)
    if config["namelist"]["fv_core_nml"].get("nudge", False):
        config = fv3kube.update_config_for_nudging(config, args.rundir)
    print(yaml.dump(config))
