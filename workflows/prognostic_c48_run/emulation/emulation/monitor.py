import logging
import os
import json
import cftime
import f90nml
import yaml
import numpy as np
import xarray as xr
from datetime import timedelta
from mpi4py import MPI
from pathlib import Path

from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner, Quantity
from .debug import print_errors


logger = logging.getLogger(__name__)

TIME_FMT = "%Y%m%d.%H%M%S"
DIMS_MAP = {
    1: ["sample"],
    2: ["sample", "z"],
}

# Initialized later from within functions to print errors
VAR_META_PATH = None
SAVE_NC = None
SAVE_ZARR = None
OUTPUT_FREQ_SEC = None
INITIAL_TIME = None
MONITOR = None


def _bool_from_str(value: str):
    affirmatives = ["y", "yes", "true"]
    negatives = ["n", "no", "false"]

    if value.lower() in affirmatives:
        return True
    elif value.lower() in negatives:
        return False
    else:
        raise ValueError(
            f"Unrecognized value encountered in boolean conversion: {value}"
        )


@print_errors
def _load_environment_vars_into_global():
    global VAR_META_PATH
    global SAVE_NC
    global SAVE_ZARR
    global OUTPUT_FREQ_SEC

    cwd = os.getcwd()
    logger.debug(f"Current working directory: {cwd}")

    VAR_META_PATH = os.environ["VAR_META_PATH"]
    SAVE_NC = _bool_from_str(os.environ.get("SAVE_NC", "True"))
    SAVE_ZARR = _bool_from_str(os.environ.get("SAVE_ZARR", "True"))
    OUTPUT_FREQ_SEC = int(os.environ["OUTPUT_FREQ_SEC"])


@print_errors
def _load_nml():
    path = os.path.join(os.getcwd(), "input.nml")
    namelist = f90nml.read(path)
    logger.info(f"Loaded namelist for ZarrMonitor from {path}")
    
    return namelist


@print_errors
def _load_metadata():
    try:
        with open(str(VAR_META_PATH), "r") as f:
            variable_metadata = yaml.safe_load(f)
            logger.info(f"Loaded variable metadata from: {VAR_META_PATH}")
    except FileNotFoundError:
        variable_metadata = {}
        logger.info(f"No metadata found at: {VAR_META_PATH}")
    return variable_metadata


@print_errors
def _load_monitor(namelist):

    partitioner = CubedSpherePartitioner.from_namelist(namelist)

    output_zarr = os.path.join(os.getcwd(), "state_output.zarr")
    output_monitor = ZarrMonitor(
        output_zarr,
        partitioner,
        mpi_comm=MPI.COMM_WORLD
    )
    logger.info(f"Initialized zarr monitor at: {output_zarr}")
    return output_monitor


@print_errors
def _get_timestep(namelist):
    return int(namelist["coupler_nml"]["dt_atmos"])


_load_environment_vars_into_global()
namelist = _load_nml()
DT_SEC = _get_timestep(namelist)
VAR_METADATA = _load_metadata()

if SAVE_ZARR:
    MONITOR = _load_monitor(namelist)


def print_rank(state):
    logger.info(MPI.COMM_WORLD.Get_rank())


def _remove_io_suffix(key: str):
    if key.endswith("_input"):
        var_key = key[:-6]
        logger.debug(f"Removed _input with result {var_key} for metadata mapping")
    elif key.endswith("_output"):
        var_key = key[:-7]
        logger.debug(f"Removed _output with result {var_key} for metadata mapping")
    else:
        var_key = key
    
    return var_key


def _get_attrs(key: str):
    key = _remove_io_suffix(key)
    if key in VAR_METADATA:
        meta = dict(**VAR_METADATA[key])
        meta = {k: json.dumps(v) for k,v in meta.items()}
    else:
        logger.debug(f"No metadata found for {key}... skipping")
        meta = {}

    return meta


def _convert_to_quantities(state):

    quantities = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key)
        units = attrs.pop("units", "unknown")
        quantities[key] = Quantity(data_t, dims, units)
        # Access to private member could break TODO: Quantity kwarg for attrs?
        quantities[key]._attrs.update(attrs)

    return quantities


def _convert_to_xr_dataset(state):

    dataset = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key)
        attrs["units"] = attrs.pop("units", "unknown")
        dataset[key] = xr.DataArray(data_t, dims=dims, attrs=attrs)

    return xr.Dataset(dataset)


def _translate_time(time):
    year = time[0]
    month = time[1]
    day = time[2]
    hour = time[4]
    min = time[5]
    datetime = cftime.DatetimeJulian(year, month, day, hour, min)
    logger.debug(f"Translated input time: {datetime}")

    return datetime


def _store_interval_check(time):

    global INITIAL_TIME

    if INITIAL_TIME is None:
        INITIAL_TIME = time

    # add increment since we are in the middle of timestep
    increment = timedelta(seconds=DT_SEC)
    elapsed = (time + increment) - INITIAL_TIME

    logger.debug(f"Time elapsed after increment: {elapsed}")
    logger.debug(f"Output frequency modulus: {elapsed.seconds % OUTPUT_FREQ_SEC}")

    return elapsed.seconds % OUTPUT_FREQ_SEC == 0


def _store_netcdf(state, time):
    nc_dump_path = os.path.join(os.getcwd(), "netcdf_output")
    if not os.path.exists(nc_dump_path):
        os.makedirs(nc_dump_path, exist_ok=True)

    logger.debug(f"Model fields: {list(state.keys())}")
    logger.info(f"Storing state to netcdf on rank {MPI.COMM_WORLD.Get_rank()}")
    ds = _convert_to_xr_dataset(state)
    rank = MPI.COMM_WORLD.Get_rank()
    coords = {"time": time, "tile": rank}
    ds = ds.assign_coords(coords)
    filename = f"state_{time.strftime(TIME_FMT)}_{rank}.nc"
    out_path = os.path.join(nc_dump_path, filename)
    ds.to_netcdf(out_path)


def _store_zarr(state, time):

    logger.info(f"Storing zarr model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    state = _convert_to_quantities(state)
    state["time"] = time
    MONITOR.store(state)


@print_errors
def store(state):

    state = dict(**state)
    time = _translate_time(state.pop("model_time"))

    if _store_interval_check(time):

        if SAVE_ZARR:
            _store_zarr(state, time)
        else:
            logger.debug(f"Zarr saving disabled by environment variable SAVE_ZARR={SAVE_ZARR}")

        if SAVE_NC:
            _store_netcdf(state, time)
        else:
            logger.debug(f"NetCDF saving disabled by environment variable SAVE_NC={SAVE_NC}")
