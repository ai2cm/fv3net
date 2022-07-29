import argparse
import atexit

from fv3fit._shared.taper_function import taper_decay, taper_mask, taper_ramp
from fv3fit._shared.models import OutOfSampleModel
from fv3net.diagnostics.offline._helpers import copy_outputs
import os
import stat
import tempfile
import uuid
import yaml


def write_oos_config(
    base_path, nd_path, cutoff, tapering_function, config_dir_path, temp_path
):
    oos_config = {
        "base_model_path": base_path,
        "novelty_detector_path": nd_path,
        "cutoff": cutoff,
    }
    if tapering_function is not None:
        oos_config["tapering_function"] = tapering_function
    os.makedirs(temp_path)
    with open(os.path.join(temp_path, OutOfSampleModel._CONFIG_FILENAME), "w") as f:
        yaml.safe_dump(oos_config, f)
    name_string = "out_of_sample"
    with open(os.path.join(temp_path, "name"), "w") as f:
        f.write(name_string)
    copy_outputs(temp_path, config_dir_path)


def write_prognostic_run_config(
    prognostic_run_config_template_path, prognostic_run_config_path, model_paths
):
    with open(prognostic_run_config_template_path, "r") as f:
        prognostic_run_config = yaml.safe_load(f)
    prognostic_run_config["scikit_learn"]["model"] = model_paths
    with open(prognostic_run_config_path, "w") as f:
        yaml.safe_dump(prognostic_run_config, f)


def write_run_str(run_str, launch_destination):
    run_path = os.path.join(launch_destination, "run.sh")
    with open(run_path, "w") as f:
        f.write(run_str)
    os.chmod(run_path, stat.S_IRWXU)


def make_argo_submit_command(
    bucket, project, experiment, trial: int, prognostic_run_config_path, segment_count
) -> str:
    trial_string = f"trial{trial}"
    tag = f"{experiment}-{trial}"
    job_name = f"prognostic-run-{uuid.uuid4().hex}"
    argo_submit_str = (
        "argo submit --from workflowtemplate/prognostic-run \\\n"
        + f"-p bucket={bucket} \\\n"
        + f"-p project={project} \\\n"
        + f"-p tag={tag} \\\n"
        + f'-p config="$(< {prognostic_run_config_path})" \\\n'
        + f"-p segment-count={segment_count} \\\n"
        + "-p memory='23Gi' \\\n"
        + "-p cpu='24' \\\n"
        + "-p online-diags-flags='--verification  1yr_pire_postspinup --n-jobs 5' \\\n"
        + f"--name '{job_name}' \\\n"
        + f"--labels 'project={project},experiment={experiment},"
        + f"trial={trial_string}' \n\n"
    )
    output_path = (
        f"gs://{bucket}/{project}/TODAY'S_DATE/"
        + f"{experiment}-{trial_string}/fv3gfs_run"
    )
    return argo_submit_str, job_name, output_path


def _cleanup_temp_dir(temp_dir):
    temp_dir.cleanup()


def get_tapering_string(tapering_function: dict):
    if tapering_function["name"] == taper_mask.__name__:
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
    temp_dir = tempfile.TemporaryDirectory()
    atexit.register(_cleanup_temp_dir, temp_dir)

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

    tendency_paths = {
        "tq": all_experiment_config["base_model"]["tq_tendencies_path"],
        "wind": all_experiment_config["base_model"]["wind_tendencies_path"],
    }
    n = 0
    run_str = "#!/bin/bash\n\nset -e\n\n"
    for novelty_detector in all_experiment_config["novelty_detectors"]:
        nd_name = novelty_detector["nd_name"]
        nd_path = novelty_detector["nd_path"]
        for param in novelty_detector["params"]:
            cutoff = param["cutoff"]
            if "tapering_function" in param:
                tapering_function = param["tapering_function"]
            else:
                tapering_function = None
            tapering_string = get_tapering_string(tapering_function)
            n += 1
            model_config_paths = []
            for tendency, tendency_path in tendency_paths.items():
                tendency_id = f"{nd_name}-{cutoff}-{tapering_string}-{tendency}"
                model_config_path = os.path.join(model_config_dir_path, tendency_id)
                temp_config_path = os.path.join(temp_dir.name, tendency_id)
                model_config_paths.append(model_config_path)
                write_oos_config(
                    tendency_path,
                    nd_path,
                    cutoff,
                    tapering_function,
                    model_config_dir_path,
                    temp_config_path,
                )
            prognostic_config_path_dir_name = f"{nd_name}-{cutoff}-{tapering_string}"
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
            experiment = f"{experiment_base}-{nd_name}-{cutoff}"
            argo_submit_str, job_name, output_path = make_argo_submit_command(
                bucket,
                project,
                experiment,
                trial,
                os.path.join(prognostic_config_path_dir_name, "prognostic-run.yaml"),
                segments,
            )
            run_str += argo_submit_str
            print(f"Experiment #{n}: {nd_name} with cutoff={cutoff}")
            print(f"argo job: {job_name}")
            print(f"output location: {output_path}")
            print(f"config path: {prognostic_run_config_path}")
    write_run_str(run_str, launch_destination)
    print(f"run path: {os.path.join(launch_destination, 'run.sh')}")


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
