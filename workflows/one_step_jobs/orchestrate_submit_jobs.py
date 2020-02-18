import argparse
import os
import logging
import subprocess
import yaml
from pathlib import Path

from fv3net.pipelines.kube_jobs import one_step

PWD = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIRECTORY_NAME = "one_step_config"


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_url",
        type=str,
        help="Remote url to initial conditions. Initial conditions are assumed to be "
        "stored as INPUT_URL/{timestamp}/{timestamp}.{restart_category}.tile*.nc",
    )
    parser.add_argument(
        "output_url",
        type=str,
        help="Remote url where model output will be saved."
    )
    parser.add_argument(
        "one_step_yaml",
        type=str,
        help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "experiment_label",
        type=str,
        help="Label to group the individual Kubernetes jobs with",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        required=False,
        action="store_true",
        help="Overwrite successful timesteps in OUTPUT_URL.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        required=False,
        help="Number of timesteps to process. By default all timesteps "
        "found in INPUT_URL for which successful runs do not exist in "
        "OUTPUT_URL will be processed. Useful for testing.",
    )
    parser.add_argument(
        "--config-url",
        type=str,
        required=False,
        help="Storage path for job configuration files"
    )

    return parser


def wait_for_complete(experiment_label):

    wait_script = os.path.join(PWD, "wait_until_complete.sh")
    subprocess.check_call([wait_script, str(experiment_label)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = _create_arg_parser()
    args = parser.parse_args()

    with open(args.one_step_yaml) as file:
        one_step_config = yaml.load(file, Loader=yaml.FullLoader)
    workflow_name = Path(args.one_step_yaml).with_suffix("").name

    if not args.config_url:
        config_url = os.path.join(args.output_url, CONFIG_DIRECTORY_NAME)
    else:
        config_url = args.config_url

    timestep_list = one_step.timesteps_to_process(
        args.input_url, args.output_url, args.n_steps, args.overwrite
    )

    local_vgrid_file = os.path.join(PWD, one_step.VERTICAL_GRID_FILENAME)
    one_step.submit_jobs(
        timestep_list, workflow_name, one_step_config, args.input_url, args.output_url,
        config_url, experiment_label=args.experiment_label,
        local_vertical_grid_file=local_vgrid_file
    )

    wait_for_complete(args.experiment_label)
