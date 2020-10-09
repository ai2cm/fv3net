from datetime import datetime, timedelta
import os
import sys
import yaml
import argparse

import vcm
import fv3config
import fv3kube


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudge-to-obs run"
    )
    parser.add_argument("config", type=str, help="base yaml file to configure")
    parser.add_argument(
        "--nudge-url",
        type=str,
        help="path to GFS analysis files",
        default="gs://vcm-ml-data/2019-12-02-year-2016-T85-nudging-data",
    )
    parser.add_argument(
        "--segment-count",
        type=int,
        help="number of segments for run-fv3gfs. Used for output times.",
        default=1,
    )
    return parser.parse_args()


def get_output_times(current_date, duration, interval):
    start_time = datetime(*current_date)
    output_times = []
    current_time = start_time + interval  # first output is after one interval
    while current_time <= start_time + duration:
        output_times.append(vcm.encode_time(current_time))
        current_time += interval
    return output_times


def test_get_output_times():
    current_date = [2016, 8, 1, 0, 0, 0]
    duration = timedelta(hours=3)
    interval = timedelta(hours=1)
    expected = ["20160801.010000", "20160801.020000", "20160801.030000"]
    assert get_output_times(current_date, duration, interval) == expected


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)


if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config = fv3kube.get_base_fv3config(user_config["base_version"])
    config = fv3kube.merge_fv3config_overlays(config, user_config)

    current_date = config["namelist"]["coupler_nml"]["current_date"]
    output_interval = timedelta(hours=config["namelist"]["atmos_model_nml"]["fhout"])
    run_duration = fv3config.get_run_duration(config)
    output_times = get_output_times(
        current_date, args.segment_count * run_duration, output_interval
    )

    config = fv3kube.merge_fv3config_overlays(
        config, {"runfile_output": {"output_times": output_times}},
    )

    if config["namelist"]["fv_core_nml"].get("nudge", False):
        if args.nudge_url.startswith("gs://"):
            copy_method = "copy"
        else:
            copy_method = "link"

        nudge_overlay = fv3kube.enable_nudge_to_observations(
            run_duration,
            current_date,
            nudge_url=args.nudge_url,
            copy_method=copy_method,
        )
        config = fv3kube.merge_fv3config_overlays(config, nudge_overlay)

    print(yaml.dump(config))
