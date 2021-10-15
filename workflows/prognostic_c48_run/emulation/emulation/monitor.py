import logging
import os
import json
from typing import Mapping
import cftime
import f90nml
import yaml
import numpy as np
import xarray as xr
from datetime import timedelta
from mpi4py import MPI

from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner, Quantity
from .debug import print_errors


logger = logging.getLogger(__name__)

TIME_FMT = "%Y%m%d.%H%M%S"
DIMS_MAP = {
    1: ["sample"],
    2: ["sample", "z"],
}


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
def _load_nml():
    path = os.path.join(os.getcwd(), "input.nml")
    namelist = f90nml.read(path)
    logger.info(f"Loaded namelist for ZarrMonitor from {path}")

    return namelist


@print_errors
def _load_metadata(path: str):
    try:
        with open(str(path), "r") as f:
            variable_metadata = yaml.safe_load(f)
            logger.info(f"Loaded variable metadata from: {path}")
    except FileNotFoundError:
        variable_metadata = {}
        logger.info(f"No metadata found at: {path}")

    return variable_metadata


@print_errors
def _load_monitor(namelist):

    partitioner = CubedSpherePartitioner.from_namelist(namelist)

    output_zarr = os.path.join(os.getcwd(), "state_output.zarr")
    output_monitor = ZarrMonitor(output_zarr, partitioner, mpi_comm=MPI.COMM_WORLD)
    logger.info(f"Initialized zarr monitor at: {output_zarr}")
    return output_monitor


@print_errors
def _get_timestep(namelist):
    return int(namelist["coupler_nml"]["dt_atmos"])


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
        meta = {k: json.dumps(v) for k, v in meta.items()}
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


def _create_nc_path():

    nc_dump_path = os.path.join(os.getcwd(), "netcdf_output")
    if not os.path.exists(nc_dump_path):
        os.makedirs(nc_dump_path, exist_ok=True)

    return nc_dump_path


def _store_netcdf(state, time, nc_dump_path):

    logger.debug(f"Model fields: {list(state.keys())}")
    logger.info(f"Storing state to netcdf on rank {MPI.COMM_WORLD.Get_rank()}")
    ds = _convert_to_xr_dataset(state)
    rank = MPI.COMM_WORLD.Get_rank()
    coords = {"time": time, "tile": rank}
    ds = ds.assign_coords(coords)
    filename = f"state_{time.strftime(TIME_FMT)}_{rank}.nc"
    out_path = os.path.join(nc_dump_path, filename)
    ds.to_netcdf(out_path)


def _store_zarr(state, time, monitor):

    logger.info(f"Storing zarr model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    state = _convert_to_quantities(state)
    state["time"] = time
    monitor.store(state)


class Config:
    """
    Singleton class for configuring and using storage
    """

    def __init__(self, var_meta_path: str, output_freq_sec: int, save_nc: bool = True, save_zarr: bool = True):
        self.var_meta_path = var_meta_path
        self.output_freq_sec = output_freq_sec
        self.save_nc = save_nc
        self.save_zarr = save_zarr

        self.namelist = _load_nml()

        self.initial_time = None
        self.dt_sec = _get_timestep(self.namelist)
        self.metadata = _load_metadata(self.var_meta_path)

        if self.save_zarr:
            self.monitor = _load_monitor(self.namelist)
        else:
            self.monitor = None

        if self.save_nc:
            self.nc_dump_path = _create_nc_path()

    @classmethod
    def from_environ(cls, d: Mapping):

        cwd = os.getcwd()
        logger.debug(f"Current working directory: {cwd}")

        var_meta_path = str(d["VAR_META_PATH"])
        output_freq_sec = int(d["OUTPUT_FREQ_SEC"])
        save_nc = _bool_from_str(d.get("SAVE_NC", "True"))
        save_zarr = _bool_from_str(d.get("SAVE_ZARR", "True"))

        return cls(var_meta_path, output_freq_sec, save_nc=save_nc, save_zarr=save_zarr)

    def _store_interval_check(self, time):

        # add increment since we are in the middle of timestep
        increment = timedelta(seconds=self.dt_sec)
        elapsed = (time + increment) - self.initial_time

        logger.debug(f"Time elapsed after increment: {elapsed}")
        logger.debug(f"Output frequency modulus: {elapsed.seconds % self.output_freq_sec}")

        return elapsed.seconds % self.output_freq_sec == 0

    def store(self, state):

        state = dict(**state)
        time = _translate_time(state.pop("model_time"))

        if self.initial_time is None:
            self.initial_time = time

        if self._store_interval_check(time):

            logger.debug("Store flags: save_zarr={self.save_zarr}, save_nc={self.save_nc}")

            if self.save_zarr:
                _store_zarr(state, time, self.monitor)

            if self.save_nc:
                _store_netcdf(state, time, self.nc_dump_path)


config = Config.from_environ(os.environ)
store = config.store
