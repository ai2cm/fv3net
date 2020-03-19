import logging
import os
import zarr
import fsspec
import numpy as np
import uuid
import yaml
import re
from copy import deepcopy
from multiprocessing import Pool
from typing import List, Tuple


import fv3config
from . import utils
from ..common import list_timesteps, subsample_timesteps_at_interval
from vcm.cloud.fsspec import get_fs

STDOUT_FILENAME = "stdout.log"
VERTICAL_GRID_FILENAME = "fv_core.res.nc"
DIAG_TABLE_NAME = "diag_table"

SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = SECONDS_IN_MINUTE * 60
SECONDS_IN_DAY = SECONDS_IN_HOUR * 24

KUBERNETES_CONFIG_DEFAULT = {
    "docker_image": "us.gcr.io/vcm-ml/fv3gfs-python",
    "cpu_count": 6,
    "gcp_secret": "gcp-key",
    "image_pull_policy": "Always",
}

logger = logging.getLogger(__name__)


def _compute_chunks(shape, chunks):
    return tuple(
        size if chunk == -1 else chunk
        for size, chunk in zip(shape, chunks)
    )


def _get_schema(shape=(3, 15, 6, 79, 48, 48)):
    variables = ["air_temperature" , "specific_humidity", "pressure_thickness_of_atmospheric_layer"]
    dims_scalar = ['step', 'lead_time', 'tile', 'z', 'y', 'x']
    chunks_scalar = _compute_chunks(shape, [-1, 1, -1, -1, -1, -1])
    DTYPE = np.float32
    scalar_schema = {"dims": dims_scalar, "chunks": chunks_scalar, "dtype": DTYPE, "shape": shape}
    return {key: scalar_schema for key in variables}


def _init_group_with_schema(group, schemas, timesteps):
    for name, schema in schemas.items():
        shape = (len(timesteps),) + schema['shape']
        chunks = (1,) + schema['chunks']
        array = group.empty(name, shape=shape, chunks=chunks, dtype=schema['dtype'])
        array.attrs.update({
            '_ARRAY_DIMENSIONS': ['initial_time'] + schema['dims']
        })


def create_zarr_store(timesteps, output_url):
    schemas = _get_schema()
    mapper = fsspec.get_mapper(output_url)
    group = zarr.open_group(mapper, mode='w')
    _init_group_with_schema(group, schemas, timesteps)
    

def timesteps_to_process(
    input_url: str,
    output_url: str,
    n_steps: int,
    overwrite: bool,
    subsample_frequency: int = None,
) -> List[str]:
    """
    Return list of timesteps left to process. This is all the timesteps in
    input_url at the subsampling frequency (if specified) minus the
    successfully completed timesteps in output_url. List is
    also limited to a length of n_steps (which can be None, i.e. no limit)
    """
    rundirs_url = os.path.join(output_url)
    to_do = list_timesteps(input_url)
    if subsample_frequency is not None:
        to_do = subsample_timesteps_at_interval(to_do, subsample_frequency)
    done = check_runs_complete(rundirs_url)
    if overwrite:
        _delete_logs_of_done_timesteps(output_url, done)
        done = []
    timestep_list = sorted(list(set(to_do) - set(done)))[:n_steps]

    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {min(len(done), len(to_do))}")
    logger.info(f"Number of times to process: {len(timestep_list)}")
    return timestep_list


def _delete_logs_of_done_timesteps(output_url: str, timesteps: List[str]):
    fs = get_fs(output_url)
    for timestep in timesteps:
        rundir_url = os.path.join(output_url, timestep)
        fs.rm(os.path.join(rundir_url, STDOUT_FILENAME))


# Run Completion Checks
def check_runs_complete(rundir_url: str):
    """Checks for existence of stdout.log and tail of said logfile to see if run
    successfully completed"""
    # TODO: Ideally this would check some sort of model exit code
    timestep_check_args = [
        (curr_timestep, os.path.join(rundir_url, curr_timestep, STDOUT_FILENAME))
        for curr_timestep in list_timesteps(rundir_url)
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
    if get_fs(logfile_path).exists(logfile_path) and _check_log_tail(logfile_path):
        return timestep
    else:
        return None


def _check_log_tail(gcs_log_file: str) -> bool:
    """
    Check the tail of an FV3GFS stdout log for output we expect upon successful
    completion.
    """

    output_timing_header = [
        "tmin",
        "tmax",
        "tavg",
        "tstd",
        "tfrac",
        "grain",
        "pemin",
        "pemax\n",
    ]
    output_timing_row_lead = [
        "Total",
        "Initialization",
        "FV",
        "FV",
        "FV",
        "GFS",
        "GFS",
        "GFS",
        "Dynamics",
        "Dynamics",
        "FV3",
        "Main",
        "Termination",
    ]
    with fsspec.open(gcs_log_file, "r") as f:
        log_output = f.readlines()[-15:]
    headers = [word for word in log_output[0].split(" ") if word]
    top_check = _check_header_categories(output_timing_header, headers)
    row_headers = [line.split(" ")[0] for line in log_output[1:-1]]
    row_check = _check_header_categories(output_timing_row_lead, row_headers)
    return top_check and row_check


def _check_header_categories(
    target_categories: List[str], source_categories: List[str]
) -> bool:
    if not source_categories:
        return False
    elif len(source_categories) != len(target_categories):
        return False
    for i, target in enumerate(target_categories):
        if target != source_categories[i]:
            return False
    return True


# Configuration Handling





def _get_vertical_grid_asset(config_url: str) -> dict:
    """Get asset corresponding to fv_core.res.nc which describes vertical coordinate"""
    return fv3config.get_asset_dict(
        config_url,
        VERTICAL_GRID_FILENAME,
        target_location="INPUT",
        target_name=VERTICAL_GRID_FILENAME,
    )


def _get_experiment_name(first_tag: str, second_tag: str) -> str:
    """Return unique experiment name with only alphanumeric, hyphen or dot"""
    uuid_short = str(uuid.uuid4())[:8]
    exp_name = f"one-step.{first_tag}.{second_tag}.{uuid_short}"
    return re.sub(r"[^a-zA-Z0-9.\-]", "", exp_name)


def _update_config(
    base_config_version: str,
    user_model_config: dict,
    output_url
) -> Tuple[dict]:
    """
    Update kubernetes and fv3 configurations with user inputs
    to prepare for fv3gfs one-step runs.
    """
    base_model_config = utils.get_base_fv3config(base_config_version)
    model_config = utils.update_nested_dict(base_model_config, user_model_config)
    model_config = fv3config.enable_restart(model_config)

    model_config["initial_conditions"] = [_get_vertical_grid_asset(output_url)]
    model_config['diag_table'] = os.path.join(output_url, DIAG_TABLE_NAME)

    return model_config


def submit_jobs(
    image,
    diag_table,
    vertical_grid,
    timestep_list: List[str],
    workflow_name: str,
    one_step_config: dict,
    input_url: str,
    output_url: str,
    config_url: str,
    base_config_version,
    job_labels=None,
) -> None:
    """Submit one-step job for all timesteps in timestep_list"""

    # setup URLS
    zarr_url = os.path.join(output_url, "big.zarr")
    diag_url = os.path.join(output_url, DIAG_TABLE_NAME)
    grid_url = os.path.join(output_url, VERTICAL_GRID_FILENAME)
    config_template_url = os.path.join(output_url, "fv3config.yml")

    # make the template
    config = _update_config(
        base_config_version, one_step_config['fv3config'], output_url)

    # intialize output directory
    create_zarr_store(timestep_list, zarr_url)
    fs = get_fs(output_url)
    fs.put(diag_table, diag_url)
    fs.put(vertical_grid, grid_url)

    with fsspec.open(config_template_url, "w") as f:
        yaml.dump(config, f)

    kube_config = one_step_config['kubernetes']
    image = kube_config.pop('docker_image')
    kube_config.pop('runfile')

    for k, timestep in enumerate(timestep_list):
        command = ["python", "run_one_step.py", input_url, output_url, timestep, str(k)]
        kube_obj = fv3config.KubernetesConfig(
            image=image,
            command=command,
            working_dir="/workflows/one_step_jobs",
            **kube_config
        )
        kube_obj.submit(namespace="default")
        logger.info(f"Submitted job for timestep {timestep}")
