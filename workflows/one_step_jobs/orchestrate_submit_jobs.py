import argparse
import os
import logging
import yaml
from pathlib import Path

from fv3net.pipelines.kube_jobs import one_step
from fv3net.pipelines.common import get_alphanumeric_unique_tag, list_timesteps

PWD = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIRECTORY_NAME = "one_step_config"

RUNFILE = os.path.join(PWD, "runfile.py")


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_url",
        type=str,
        help="Remote url to initial conditions. Initial conditions are assumed to be "
        "stored as INPUT_URL/{timestamp}/{timestamp}.{restart_category}.tile*.nc",
    )
    parser.add_argument(
        "one_step_yaml", type=str, help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "docker_image",
        type=str,
        help="Docker image to use for performing the one-step FV3GFS runs.",
    )
    parser.add_argument(
        "output_url", type=str, help="Remote url where model output will be saved."
    )
    parser.add_argument(
        "--i-start",
        type=int,
        default=0,
        required=False,
        help="Index of timestep at which to start. By default starts at first "
        "timestep. Useful for testing.",
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
        help="Storage path for job configuration files",
    )
    parser.add_argument(
        "--config-version",
        type=str,
        required=False,
        default="v0.3",
        help="Default fv3config.yml version to use as the base configuration. "
        "This should be consistent with the fv3gfs-python version in the specified "
        "docker image.",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = _create_arg_parser()
    args = parser.parse_args()

    with open(args.one_step_yaml) as file:
        one_step_config = yaml.load(file, Loader=yaml.FullLoader)
    workflow_name = Path(args.one_step_yaml).with_suffix("").name
    short_id = get_alphanumeric_unique_tag(8)
    job_label = {
        "orchestrator-jobs": f"{workflow_name}-{short_id}",
        "workflow": "one_step_jobs",
    }

    if not args.config_url:
        config_url = os.path.join(args.output_url, CONFIG_DIRECTORY_NAME)
    else:
        config_url = args.config_url

    timesteps = list_timesteps(args.input_url)
    i_stop = None if args.n_steps is None else args.i_start + args.n_steps
    timesteps = timesteps[slice(args.i_start, i_stop)]

    one_step_config["kubernetes"]["runfile"] = RUNFILE
    one_step_config["kubernetes"]["docker_image"] = args.docker_image

    local_vgrid_file = os.path.join(PWD, one_step.VERTICAL_GRID_FILENAME)
    one_step.submit_jobs(
        timesteps,
        workflow_name,
        one_step_config,
        args.input_url,
        args.output_url,
        config_url,
        args.config_version,
        job_labels=job_label,
        local_vertical_grid_file=local_vgrid_file,
    )
