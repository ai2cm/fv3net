import argparse
from datetime import datetime
from fv3fit._shared.taper_function import taper_decay, taper_mask, taper_ramp
from fv3fit._shared.models import OutOfSampleModel
from fv3net.diagnostics.offline._helpers import copy_outputs
import os
import stat
import tempfile
from typing import List, Mapping, Optional, Tuple
import uuid
import yaml


def write_oos_config(
    base_path: str,
    nd_path: str,
    cutoff: float,
    tapering_function: Optional[Mapping],
    config_dir_path: str,
):
    """
    Uploads a config file and a name file to `config_dir_path` on gcloud that can be
    used to load an instance of the OutOfSampleModel class.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # creates and saves the config file in a temp directory
        oos_config = {
            "base_model_path": base_path,
            "novelty_detector_path": nd_path,
            "cutoff": cutoff,
        }
        if tapering_function is not None:
            oos_config["tapering_function"] = tapering_function

        with open(os.path.join(temp_dir, OutOfSampleModel._CONFIG_FILENAME), "w") as f:
            yaml.safe_dump(oos_config, f)

        # creates and saves the name file in a temp directory
        with open(os.path.join(temp_dir, "name"), "w") as f:
            f.write("out_of_sample")

        # moves the temp directory to the config path
        copy_outputs(temp_dir, config_dir_path)


def write_prognostic_run_config(
    prognostic_run_config_template_path: str,
    prognostic_run_config_path: str,
    model_paths: List[str],
):
    """
    Using a template at prognostic_run_config_template_path (which contains all
    information, except the scikit_learn section), writes the prognostic run config
    file with one or two ML-corrective models to the target path
    prognostic_run_config_path.
    """
    with open(prognostic_run_config_template_path, "r") as f:
        prognostic_run_config = yaml.safe_load(f)
    prognostic_run_config["scikit_learn"]["model"] = model_paths
    with open(prognostic_run_config_path, "w") as f:
        yaml.safe_dump(prognostic_run_config, f)


def write_run_str(run_str: str, launch_destination: str):
    """
    Writes a string to the `run.sh` file at launch_destination and makes it
    executable.
    """
    run_path = os.path.join(launch_destination, "run.sh")
    with open(run_path, "w") as f:
        f.write(run_str)
    os.chmod(run_path, stat.S_IRWXU)


def make_argo_submit_command(
    bucket: str,
    project: str,
    experiment: str,
    trial: int,
    prognostic_run_config_path: str,
    segment_count: int,
    submission_date: str,
) -> Tuple[str, str, str]:
    """
    Creates the run string needed to submit a single job with a specifified prognostic
    run configuration. Returns the string, the name of the argo job, and the output
    path of the dataset (with the date missing) that can used after the run terminates.
    """
    trial_string = f"trial{trial}"
    tag = f"{experiment}-{trial_string}"
    job_name = f"prognostic-run-{uuid.uuid4().hex}"
    argo_submit_str = (
        "argo submit --from workflowtemplate/prognostic-run \\\n"
        + f"-p bucket={bucket} \\\n"
        + f"-p project={project} \\\n"
        + f"-p tag={tag} \\\n"
        + f'-p config="$(< {prognostic_run_config_path})" \\\n'
        + f"-p segment-count={segment_count} \\\n"
        + "-p memory='25Gi' \\\n"
        + "-p cpu='24' \\\n"
        + "-p online-diags-flags='--verification  1yr_pire_postspinup --n-jobs 5' \\\n"
        + f"--name '{job_name}' \\\n"
        + f"--labels 'project={project},experiment={experiment},"
        + f"trial={trial_string}' \n\n"
    )
    output_path = (
        f"gs://{bucket}/{project}/{submission_date}/"
        + f"{experiment}-{trial_string}/fv3gfs_run"
    )
    return argo_submit_str, job_name, output_path


def _cleanup_temp_dir(temp_dir):
    temp_dir.cleanup()


def get_tapering_string(tapering_function: Mapping) -> str:
    """
        Creates a unique string for each tapering function that is used to
        distinguish tasks.
    """
    if tapering_function is None:
        return "default"
    elif tapering_function["name"] == taper_mask.__name__:
        taper_str = "mask"
        if "cutoff" in tapering_function:
            taper_str += f"-{tapering_function['cutoff']}"
    elif tapering_function["name"] == taper_ramp.__name__:
        taper_str = "ramp"
        if "ramp_min" in tapering_function:
            taper_str += f"-{tapering_function['ramp_min']}"
        if "ramp_max" in tapering_function:
            taper_str += f"-{tapering_function['ramp_max']}"
    elif tapering_function["name"] == taper_decay.__name__:
        taper_str = "decay"
        if "threshold" in tapering_function:
            taper_str += f"-{tapering_function['threshold']}"
        if "rate" in tapering_function:
            taper_str += f"-{tapering_function['rate']}"
    return taper_str


def prep_oos_experiments(args):
    """
    Processes a config file to create a collection of out-of-sample augmented
    simulations that can be easily launched as argo jobs.
    """
    # loads the input config file as a dictionary to parse
    with open(args.config_path, "r") as f:
        try:
            all_experiment_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    prognostic_run_config_template_path = all_experiment_config[
        "prognostic_run_config_template"
    ]
    launch_destination = all_experiment_config["launch_destination"]
    if launch_destination.startswith("~"):
        launch_destination = os.path.expanduser(launch_destination)
    model_config_dir_path = all_experiment_config["model_config_dir_path"]
    segments = all_experiment_config["segments"]
    result_destination_config = all_experiment_config["result_destination"]
    bucket = result_destination_config["bucket"]
    project = result_destination_config["project"]
    experiment_base = result_destination_config["experiment"]
    trial = result_destination_config["trial"]
    submission_date = all_experiment_config.get(
        "submission_date", datetime.now().strftime("%Y-%m-%d")
    )
    experiment_summary_path = all_experiment_config["experiment_summary_path"]

    # reads the paths to temperature/humidity and winds
    # - tq tendencies are required
    # - wind tendencies may not exist: then, use only tq tendencies
    # - wind tendencies may have one path: then, use that pair
    # - wind tendencies may have a list of paths: then, use all (tq, wind) pairs
    tendency_paths = {"tq": all_experiment_config["base_model"]["tq_tendencies_path"]}
    if "wind_tendencies_path" in all_experiment_config["base_model"]:
        tendency_paths["wind"] = [
            all_experiment_config["base_model"]["wind_tendencies_path"]
        ]
        wind_path_iterates = 1
    elif "wind_tendencies_paths" in all_experiment_config["base_model"]:
        tendency_paths["wind"] = all_experiment_config["base_model"][
            "wind_tendencies_paths"
        ]
        wind_path_iterates = len(tendency_paths["wind"])
    else:
        wind_path_iterates = 1

    # tracks the total number of submitted jobs
    n = 0
    # string to be saved to run.sh file
    run_str = "#!/bin/bash\n\nset -e\n\n"
    # file to write information
    with open(experiment_summary_path, "w") as summary_f:
        # loops over all novelty detector paths supplied
        for novelty_detector in all_experiment_config["novelty_detectors"]:
            nd_name = novelty_detector["nd_name"]
            nd_path = novelty_detector["nd_path"]
            # loops over all parameter settings for each novelty detector path
            for param in novelty_detector["params"]:
                cutoff = param["cutoff"]
                if "tapering_function" in param:
                    tapering_function = param["tapering_function"]
                else:
                    tapering_function = None
                tapering_string = get_tapering_string(tapering_function)
                model_config_paths = []
                # loops over every wind tendency
                # (only one pass if 0 or 1 wind tendency suppled)
                for i in range(wind_path_iterates):
                    n += 1
                    # loops over each tendency in a pair (tq + wind)
                    for tendency, tendency_path in tendency_paths.items():
                        if tendency == "wind":
                            tendency_path = tendency_path[i]
                        # creates out-of-sample config directory, uploads to gcloud
                        tendency_id = (
                            f"{nd_name}-{cutoff}-{tapering_string}-{tendency}-{i}"
                        )
                        model_config_path = os.path.join(
                            model_config_dir_path, tendency_id
                        )
                        model_config_paths.append(model_config_path)
                        write_oos_config(
                            tendency_path,
                            nd_path,
                            cutoff,
                            tapering_function,
                            model_config_path,
                        )

                    # creates and writes prognostic run config file
                    prognostic_config_path_dir_name = (
                        f"{nd_name}-{cutoff}-{tapering_string}-{i}"
                    )
                    prognostic_run_config_dir_path = os.path.join(
                        launch_destination, prognostic_config_path_dir_name
                    )
                    if not os.path.exists(prognostic_run_config_dir_path):
                        os.makedirs(prognostic_run_config_dir_path)
                    prognostic_run_config_path = os.path.join(
                        prognostic_run_config_dir_path, "prognostic-run.yaml"
                    )
                    write_prognostic_run_config(
                        prognostic_run_config_template_path,
                        prognostic_run_config_path,
                        model_config_paths,
                    )

                    # creates run script for individual run, appends to run string
                    experiment = (
                        f"{experiment_base}-{nd_name}-{cutoff}-{tapering_string}-{i}"
                    )
                    argo_submit_str, job_name, output_path = make_argo_submit_command(
                        bucket,
                        project,
                        experiment,
                        trial,
                        os.path.join(
                            prognostic_config_path_dir_name, "prognostic-run.yaml"
                        ),
                        segments,
                        submission_date,
                    )
                    run_str += argo_submit_str

                    # prints experiment details for user
                    summary_f.write(
                        f"Experiment #{n}: {nd_name} with cutoff={cutoff}\n"
                    )
                    summary_f.write(f"argo job: {job_name}\n")
                    summary_f.write(f"output location: {output_path}\n")
                    summary_f.write(f"config path: {prognostic_run_config_path}\n\n")

        # writes run script to target location
        write_run_str(run_str, launch_destination)
        summary_f.write(f"run path: {os.path.join(launch_destination, 'run.sh')}\n")
        print(f"Experiment summary at {experiment_summary_path}")


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", default=None, type=str, help=("Path to yaml config file.")
    )
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args = parser.parse_args()
    prep_oos_experiments(args)
