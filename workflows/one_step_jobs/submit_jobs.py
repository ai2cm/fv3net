import copy
import logging
import os
import subprocess
import uuid
from datetime import timedelta
from pathlib import Path
from typing import List
from multiprocessing import Pool
import gcsfs
import yaml
import tempfile
import fv3config

from vcm.cloud import gsutil, gcs

logger = logging.getLogger("run_jobs")

RUN_DURATION = timedelta(minutes=15)
RUN_TIMESTEP = 60  # seconds

PARENT_BUCKET = "gs://vcm-ml-data/2020-01-06-X-SHiELD-2019-12-02-restarts-and-rundirs/"
INPUT_BUCKET = PARENT_BUCKET + "restarts/C48/"
OUTPUT_BUCKET = PARENT_BUCKET + "one_step_output/C48/"
CONFIG_BUCKET = PARENT_BUCKET + "one_step_config/C48/"
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_FILES_TO_UPLOAD = ["runfile.py", "diag_table", "vertical_grid/fv_core.res.nc"]

RESTART_CATEGORIES = ["fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", "sfc_data"]
TILES = range(1, 7)

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 60 * SECONDS_IN_MINUTE


def timestep_from_url(url: str) -> str:
    return str(Path(url).name)


def list_timesteps(bucket: str) -> set:
    """Returns the unique timesteps in a bucket"""
    try:
        ls_output = subprocess.check_output(["gsutil", "ls", bucket])
    except subprocess.CalledProcessError:
        return set()
    file_list = ls_output.decode("UTF-8").strip().split()
    timesteps = map(timestep_from_url, file_list)
    return set(timesteps)


def time_list_from_timestep(timestep: str) -> List[int]:
    year = int(timestep[:4])
    month = int(timestep[4:6])
    day = int(timestep[6:8])
    hour = int(timestep[9:11])
    minute = int(timestep[11:13])
    second = int(timestep[13:15])
    return [year, month, day, hour, minute, second]


def check_runs_complete(bucket: str):
    """Checks for log and tail of output to see if run successfully incomplete"""
    # TODO: Ideally this would check some sort of model exit code

    timestep_check_args = [
        (curr_timestep, os.path.join(bucket, curr_timestep, 'stdout.log'))
        for curr_timestep in list_timesteps(bucket)
    ]

    pool = Pool(processes=16)
    complete = set(pool.map(_check_run_complete_unpacker, timestep_check_args))
    pool.close()

    if None in complete:
        complete.remove(None)

    return complete


def _check_run_complete_unpacker(arg: tuple) -> str:
    return _check_run_complete_func(*arg)


def _check_run_complete_func(timestep: str, logfile_path: str) -> str:

    if gsutil.exists(logfile_path) and _check_log_tail(logfile_path):
        return timestep
    else:
        return None


def _check_log_tail(gcs_log_file: str) -> bool:

    # Checking for specific timing headers that outputs at the end of the run

    output_timing_header = [
        'tmin', 'tmax', 'tavg', 'tstd', 'tfrac', 'grain', 'pemin', 'pemax\n'
    ]
    output_timing_row_lead = [
        'Total', 'Initialization', 'FV', 'FV', 'FV', 'GFS', 'GFS', 'GFS',
        'Dynamics', 'Dynamics', 'FV3', 'Main', 'Termination'
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        filename = Path(gcs_log_file).name
        gcs_file_blob = gcs.init_blob_from_gcs_url(gcs_log_file)
        gcs.download_blob_to_file(gcs_file_blob, tmpdir, filename)

        with open(os.path.join(tmpdir, filename), 'r') as f:
            log_output = f.readlines()[-15:]

        headers = [word for word in log_output[0].split(' ') if word]
        top_check = _check_header_categories(output_timing_header, headers)
        row_headers = [line.split(' ')[0] for line in log_output[1:-1]]
        row_check = _check_header_categories(output_timing_row_lead, row_headers)

        return top_check and row_check


def _check_header_categories(target_categories, source_categories):
    if not source_categories:
        return False
    elif len(source_categories) != len(target_categories):
        return False

    for i, target in enumerate(target_categories):
        if target != source_categories[i]:
            return False
    
    return True


def timesteps_to_process() -> List[str]:
    to_do = list_timesteps(INPUT_BUCKET)
    done = check_runs_complete(OUTPUT_BUCKET)
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    logger.info(f"Number of times to process: {len(to_do) - len(done)}")
    return sorted(list(set(to_do) - set(done)))


def turn_off_physics(config):
    new_config = copy.deepcopy(config)
    new_config["namelist"]["atmos_model_nml"]["dycore_only"] = True
    new_config["namelist"]["fv_core_nml"]["do_sat_adj"] = False
    new_config["namelist"]["gfdl_cloud_microphysics_nml"]["fast_sat_adj"] = False
    return new_config


def get_initial_condition_patch_files(timestep):
    patch_files = [
        fv3config.get_asset_dict(
            os.path.join(INPUT_BUCKET, timestep),
            f"{timestep}.{category}.tile{tile}.nc",
            target_location="INPUT",
            target_name=f"{category}.tile{tile}.nc",
        )
        for category in RESTART_CATEGORIES
        for tile in TILES
    ]
    return patch_files


def get_config(timestep):
    config = fv3config.get_default_config()
    config = fv3config.enable_restart(config)
    config["experiment_name"] = f"one-step.{timestep}.{uuid.uuid4()}"
    config["diag_table"] = os.path.join(CONFIG_BUCKET, timestep, "diag_table")
    # Following has only fv_core.res.nc. Other initial conditions are in patch_files.
    config["initial_conditions"] = os.path.join(CONFIG_BUCKET, timestep, "vertical_grid")
    config["patch_files"] = get_initial_condition_patch_files(timestep)
    config = fv3config.set_run_duration(config, RUN_DURATION)
    config["namelist"]["coupler_nml"].update(
        {
            "dt_atmos": RUN_TIMESTEP,
            "dt_ocean": RUN_TIMESTEP,
            "restart_secs": RUN_TIMESTEP,
            "current_date": time_list_from_timestep(timestep),
            "force_date_from_namelist": True,
        }
    )
    config["namelist"]["fv_core_nml"].update(
        {"external_eta": True, "npz": 79, "k_split": 1, "n_split": 1}
    )
    config = turn_off_physics(config)
    return config


def submit_jobs(timestep_list: List[str]) -> None:
    fs = gcsfs.GCSFileSystem()
    for timestep in timestep_list:
        config = get_config(timestep)
        job_name = config["experiment_name"]
        for filename in LOCAL_FILES_TO_UPLOAD:
            file_local = os.path.join(LOCAL_PARENT_DIR, filename)
            file_remote = os.path.join(CONFIG_BUCKET, timestep, filename)
            fs.put(file_local, file_remote)
        config_location = os.path.join(CONFIG_BUCKET, timestep, "fv3config.yml")
        with fs.open(config_location, "w") as config_file:
            config_file.write(yaml.dump(config))
        outdir = os.path.join(OUTPUT_BUCKET, timestep)
        fv3config.run_kubernetes(
            config_location,
            outdir,
            DOCKER_IMAGE,
            runfile=os.path.join(CONFIG_BUCKET, timestep, "runfile.py"),
            jobname=job_name,
            namespace="default",
            memory_gb=3.6,
            cpu_count=6,
            gcp_secret="gcp-key",
            image_pull_policy="Always",
        )
        logger.info(f"Submitted job {job_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    timestep_list = timesteps_to_process()
    submit_jobs(timestep_list[0:1])
