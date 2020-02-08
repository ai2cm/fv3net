import argparse
import os
import logging
import subprocess

import submit_utils

PWD = os.path.dirname(os.path.abspath(__file__))
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(PWD, "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(PWD, "diag_table")


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_gcs_path",
        type=str,
        required=True,
        help="Source input restart files for one-step simulations",
    )
    parser.add_argument(
        "output_gcs_path",
        type=str,
        required=True,
        help="Output destination for one-step configurations and simulation output"
    )
    parser.add_argument(
        "experiment_label",
        type=str,
        required=True,
        help="Label to group the individual Kubernetes jobs with",
    )
    parser.add_argument(
        "--config-bucket",
        type=str,
        required=False,
        help="Storage path for job configuration files"
    )
    parser.add_argument(
        "--local-runfile",
        type=str,
        required=False,
        default=LOCAL_RUNFILE,
        help="Path to local runfile for performing a simulation using an FV3GFS Docker image",
    )
    parser.add_argument(
        "--local-diag-table",
        type=str,
        required=False,
        default=LOCAL_DIAG_TABLE,
        help="Path to local diag table file for the FV3GFS run",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        required=False,
        default=DOCKER_IMAGE,
        help="Docker image to pull for the kubernetes job",
    )

    return parser


def wait_for_complete(experiment_label):

    wait_script = os.path.join(PWD, "wait_until_comletion.sh")
    subprocess.check_call([wait_script, str(experiment_label)])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = _create_arg_parser()
    args = parser.parse_args()

    if not args.config_bucket:
        config_bucket = args.output_gcs_path + "/config"
    else:
        config_bucket = args.config_bucket

    timestep_list = submit_utils.timesteps_to_process(args.input_gcs_path, args.output_gcs_path)
    submit_utils.submit_jobs(
        timestep_list, config_bucket, args.local_runfile,
        args.local_diag_table, args.docker_image, args.output_gcs_path,
        experiment_label=args.experiment_label
    )

    wait_for_complete(args.experiment_label)
