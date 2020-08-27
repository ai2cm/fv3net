from datetime import datetime, timedelta
import os
import sys
import yaml
import argparse

import fv3config
import fv3kube
import vcm


def parse_args():
    parser = argparse.ArgumentParser(
        description="prepare fv3config yaml file for nudge-to-obs run"
    )
    parser.add_argument("config", type=str, help="base yaml file to configure")
    parser.add_argument(
        "--segment-count", type=int, help="number of run-fv3gfs segmetns", default=1
    )
    return parser.parse_args()


def get_output_times(config, segment_count):
    start_time = datetime(*config["namelist"]["coupler_nml"]["current_date"])
    output_frequency = timedelta(hours=config["namelist"]["atmos_model_nml"]["fhout"])
    duration = segment_count * fv3config.get_run_duration(config)
    output_times = []
    current_time = start_time + output_frequency  # first output is after one interval
    while current_time <= start_time + duration:
        output_times.append(vcm.encode_time(current_time))
        current_time += output_frequency
    return output_times


def test_get_output_times():
    test_config = {
        "namelist": {
            "coupler_nml": {
                "current_date": [2016, 8, 1, 0, 0, 0],
                "hours": 3,
                "years": 0,
                "months": 0,
                "days": 0,
                "minutes": 0,
                "seconds": 0,
            },
            "atmos_model_nml": {"fhout": 1.0},
        }
    }
    expected_one_segment = ["20160801.010000", "20160801.020000", "20160801.030000"]
    expected_two_segments = [
        "20160801.010000",
        "20160801.020000",
        "20160801.030000",
        "20160801.040000",
        "20160801.050000",
        "20160801.060000",
    ]
    assert get_output_times(test_config, 1) == expected_one_segment
    assert get_output_times(test_config, 2) == expected_two_segments


FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(FILE_DIR)


if __name__ == "__main__":

    args = parse_args()

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config = fv3kube.get_base_fv3config(user_config["base_version"])
    config = vcm.update_nested_dict(config, user_config)
    output_times = get_output_times(config, args.segment_count)
    config = vcm.update_nested_dict(
        config, {"runfile_output": {"output_times": output_times}},
    )

    if config["namelist"]["fv_core_nml"].get("nudge", False):
        config = fv3kube.enable_nudge_to_observations(
            config, segment_count=args.segment_count, link=True
        )

    print(yaml.dump(config))
