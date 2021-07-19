import logging
import os
import cftime
import f90nml
import yaml
import numpy as np
from mpi4py import MPI

from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner, Quantity
from .debug import print_errors


logger = logging.getLogger(__name__)

# TODO: Switch from env_var config to something compatible with ARGO
NML_PATH = os.environ.get("INPUT_NML_PATH")
VAR_META_PATH = os.environ.get("VARIABLE_METADATA_YAML")
DUMP_PATH = os.environ.get("STATE_DUMP_PATH")
DIMS_MAP = {
    1: ["horizontal_dimension"],
    2: ["lev", "horizontal_dimension"],
    3: ["tracer_number", "lev", "horizontal_dimension"]
}


@print_errors
def _load_nml():
    namelist = f90nml.read(NML_PATH)
    logger.info(f"Loaded namelist for ZarrMonitor from {NML_PATH}")
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
    output_zarr = os.path.join(str(DUMP_PATH), "state_output.zarr")
    output_monitor = ZarrMonitor(
        output_zarr,
        partitioner,
        mpi_comm=MPI.COMM_WORLD
    )
    logger.info(f"Initialized zarr monitor at: {output_zarr}")
    return output_monitor


_namelist = _load_nml()
_variable_metadata = _load_metadata()
_output_monitor = _load_monitor(_namelist)


def print_rank(state):
    logger.info(MPI.COMM_WORLD.Get_rank())


def _adjust_vert_plus_one(data, dims):
    logger.debug("Found vertical layers plus 1, adjusting dim name")
    dim_idx = dims.index("lev")
    dims = list(dims)
    if data.shape[dim_idx] == 80:
        dims[dim_idx] = "lev_plus_one"
    return dims


def _get_attrs(key):
    if key in _variable_metadata:
        var_key = key
    elif key.split("_")[0] in _variable_metadata:
        # var_input_... style
        var_key = key.split("_")[0]
        logger.debug(f"Reduced key to {var_key} for metadata mapping")
    else:
        var_key = None
        logger.debug(f"No metadata found for {key}... skipping")
        return {}

    return dict(**_variable_metadata[var_key])


def _convert_to_quantites(state):

    quantities = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        dims = DIMS_MAP[data.ndim]
        if "lev" in dims:
            dims = _adjust_vert_plus_one(data, dims)
        attrs = _get_attrs(key)
        units = attrs.pop("units", "units")
        quantities[key] = Quantity(data, dims, units)
        # Access to private member could break TODO: Quantity kwarg for attrs?
        quantities[key]._attrs.update(attrs)

    return quantities


def _translate_time(time):
    year = time[0]
    month = time[1]
    day = time[2]
    hour = time[4]
    min = time[5]
    datetime = cftime.DatetimeJulian(year, month, day, hour, min)
    logger.debug(f"Translated time: {datetime}")

    return datetime


@print_errors
def store(state):
    logger.info(f"Storing model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    time = _translate_time(state.pop("model_time"))
    state = _convert_to_quantites(state)
    state["time"] = time
    _output_monitor.store(state)
