import argparse
import os
import logging
import yaml
from pathlib import Path

from fv3net.pipelines.kube_jobs import one_step
from fv3net.pipelines.kube_jobs.utils import wait_for_kubejob_complete
from fv3net.pipelines.common import get_unique_tag

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
        "output_url", type=str, help="Remote url where model output will be saved."
    )
    parser.add_argument(
        "one_step_yaml", type=str, help="Path to local run configuration yaml.",
    )
    parser.add_argument(
        "experiment_label",
        type=str,
        help="Label to group the individual Kubernetes jobs with",
    )
    parser.add_argument(
        "docker_image",
        type=str,
        help="Docker image to use for performing the one-step FV3GFS runs.",
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
        help="Storage path for job configuration files",
    )

    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = _create_arg_parser()
    args = parser.parse_args()

    with open(args.one_step_yaml) as file:
        one_step_config = yaml.load(file, Loader=yaml.FullLoader)
    workflow_name = Path(args.one_step_yaml).with_suffix("").name
    short_id = get_unique_tag(8)
    job_label = {'orchestrator-jobs': f"{workflow_name}-{short_id}"}

    if not args.config_url:
        config_url = os.path.join(args.output_url, CONFIG_DIRECTORY_NAME)
    else:
        config_url = args.config_url

    timestep_list = one_step.timesteps_to_process(
        args.input_url, args.output_url, args.n_steps, args.overwrite
    )

    one_step_config["kubernetes"]["docker_image"] = args.docker_image

    local_vgrid_file = os.path.join(PWD, one_step.VERTICAL_GRID_FILENAME)
    one_step.submit_jobs(
        timestep_list,
        workflow_name,
        one_step_config,
        args.input_url,
        args.output_url,
        config_url,
        job_labels=job_label,
        local_vertical_grid_file=local_vgrid_file,
    )

    wait_for_kubejob_complete(job_label)
