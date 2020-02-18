import logging
import os
import argparse
import yaml
from pathlib import Path

import submit_utils

logger = logging.getLogger("run_jobs")

RUNDIRS_DIRECTORY_NAME = "one_step_output"
CONFIG_DIRECTORY_NAME = "one_step_config"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--one-step-yaml",
        type=str,
        required=True,
        help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "--input-url",
        type=str,
        required=True,
        help="Remote url to initial conditions. Initial conditions are assumed to be "
        "stored as INPUT_URL/{timestamp}/{timestamp}.{restart_category}.tile*.nc",
    )
    parser.add_argument(
        "--output-url",
        type=str,
        required=True,
        help="Remote url where model configuration and output will be saved. "
        "Specifically, configuration files will be saved to OUTPUT_URL/"
        f"{CONFIG_DIRECTORY_NAME} and model output to OUTPUT_URL/"
        f"{RUNDIRS_DIRECTORY_NAME}",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Number of timesteps to process. By default all timesteps "
        "found in INPUT_URL for which successful runs do not exist in "
        "OUTPUT_URL will be processed. Useful for testing.",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite successful timesteps in OUTPUT_URL.",
    )
    args = parser.parse_args()

    with open(args.one_step_yaml) as file:
        one_step_config = yaml.load(file, Loader=yaml.FullLoader)
    workflow_name = Path(args.one_step_yaml).with_suffix("").name

    output_url = os.path.join(args.output_url, RUNDIRS_DIRECTORY_NAME)
    config_url = os.path.join(args.output_url, CONFIG_DIRECTORY_NAME)

    timestep_list = submit_utils.timesteps_to_process(
        args.input_url, args.output_url, args.n_steps, args.overwrite
    )
    submit_utils.submit_jobs(
        timestep_list,
        workflow_name,
        one_step_config,
        args.input_url,
        output_url,
        config_url,
    )
