import dataclasses
import logging
import os
import json
from typing import Mapping, Tuple
import cftime
import f90nml
import yaml
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from mpi4py import MPI
import tensorflow as tf
import emulation.serialize

from fv3gfs.util import ZarrMonitor, CubedSpherePartitioner, Quantity
from ..debug import print_errors
from .._typing import FortranState


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


def _get_attrs(key: str, metadata: Mapping):
    key = _remove_io_suffix(key)
    if key in metadata:
        meta = dict(**metadata[key])
        meta = {k: json.dumps(v) for k, v in meta.items()}
    else:
        logger.debug(f"No metadata found for {key}... skipping")
        meta = {}

    return meta


def _convert_to_quantities(state, metadata):

    quantities = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key, metadata)
        units = attrs.pop("units", "unknown")
        quantities[key] = Quantity(data_t, dims, units)
        # Access to private member could break TODO: Quantity kwarg for attrs?
        quantities[key]._attrs.update(attrs)

    return quantities


def _convert_to_xr_dataset(state, metadata):

    dataset = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key, metadata)
        attrs["units"] = attrs.pop("units", "unknown")
        dataset[key] = xr.DataArray(data_t, dims=dims, attrs=attrs)

    return xr.Dataset(dataset)


def _translate_time(time: Tuple[int, int, int, int, int, int]) -> cftime.DatetimeJulian:

    # list order is set by fortran from variable Model%jdat
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


def _store_netcdf(state, time, nc_dump_path, metadata):

    logger.debug(f"Model fields: {list(state.keys())}")
    logger.info(f"Storing state to netcdf on rank {MPI.COMM_WORLD.Get_rank()}")
    ds = _convert_to_xr_dataset(state, metadata)
    rank = MPI.COMM_WORLD.Get_rank()
    coords = {"time": time, "tile": rank}
    ds = ds.assign_coords(coords)
    filename = f"state_{time.strftime(TIME_FMT)}_{rank}.nc"
    out_path = os.path.join(nc_dump_path, filename)
    ds.to_netcdf(out_path)


def _store_zarr(state, time, monitor, metadata):

    logger.info(f"Storing zarr model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    state = _convert_to_quantities(state, metadata)
    state["time"] = time
    monitor.store(state)


class _TFRecordStore:

    PARSER_FILE: str = "parser.tf"
    TIME: str = "time"

    def __init__(self, root: str, rank: int):
        self.rank = rank
        self.root = root
        tf.io.gfile.makedirs(self.root)
        self._tf_writer = tf.io.TFRecordWriter(
            path=os.path.join(self.root, f"rank{self.rank}.tfrecord")
        )
        self._called = False

    def _save_parser_if_needed(self, state_tf: Mapping[str, tf.Tensor]):
        # needs the state to get the parser so cannot be run in __init__
        if not self._called and self.rank == 0:
            parser = emulation.serialize.get_parser(state_tf)
            tf.saved_model.save(parser, os.path.join(self.root, self.PARSER_FILE))
            self._called = True

    def _convert_to_tensor(
        self, time: cftime.DatetimeJulian, state: Mapping[str, np.ndarray],
    ) -> Mapping[str, tf.Tensor]:
        state_tf = {key: tf.convert_to_tensor(state[key].T) for key in state}
        time = datetime(
            time.year, time.month, time.day, time.hour, time.minute, time.second
        )
        n = max([state[key].shape[0] for key in state])
        state_tf[self.TIME] = tf.convert_to_tensor([time.isoformat()] * n)
        return state_tf

    def __call__(self, state: Mapping[str, np.ndarray], time: cftime.DatetimeJulian):
        state_tf = self._convert_to_tensor(time, state)
        self._save_parser_if_needed(state_tf)
        self._tf_writer.write(emulation.serialize.serialize_tensor_dict(state_tf))
        # need to flush after every call since there are no finalization hooks
        # in the model
        self._tf_writer.flush()


@dataclasses.dataclass
class StorageConfig:
    """Storage configuration

    Attributes:
        output_freq_sec: output frequency in seconds to save
            nc and/or zarr files at
        var_meta_path: path to variable metadata added to
            saved field attributes. If not specified no metadata
            and 'unknown' units saved with fields
        save_nc: save all state fields to netcdf, default
            is true
        save_zarr: save all statefields to zarr, default
            is true
        save_tfrecord: save all statefields to tfrecord
    """

    var_meta_path: str = ""
    output_freq_sec: int = 1
    save_nc: bool = True
    save_zarr: bool = True
    save_tfrecord: bool = False


@dataclasses.dataclass
class StorageHook:
    """Stores state to nc, zarr, or tfrecords

    Notes:
        Has same attributes as ``StorageConfig``, but contains stateful
        operations.
    """

    def __init__(
        self,
        var_meta_path: str = "",
        output_freq_sec: int = 1,
        save_nc: bool = True,
        save_zarr: bool = True,
        save_tfrecord: bool = False,
    ):
        self.name = "emulation storage monitor"
        self.var_meta_path = var_meta_path
        self.output_freq_sec = output_freq_sec
        self.save_nc = save_nc
        self.save_zarr = save_zarr
        self.save_tfrecord = save_tfrecord

        self.namelist = _load_nml()

        self.initial_time = None
        self.dt_sec = _get_timestep(self.namelist)
        self.metadata = _load_metadata(self.var_meta_path) if self.var_meta_path else {}

        if self.save_zarr:
            self.monitor = _load_monitor(self.namelist)
        else:
            self.monitor = None

        if self.save_nc:
            self.nc_dump_path = _create_nc_path()

        if self.save_tfrecord:
            rank = MPI.COMM_WORLD.Get_rank()
            self._store_tfrecord = _TFRecordStore("tfrecords", rank)

    def _store_interval_check(self, time):

        # add increment since we are in the middle of timestep
        increment = timedelta(seconds=self.dt_sec)
        elapsed = (time + increment) - self.initial_time

        logger.debug(f"Time elapsed after increment: {elapsed}")
        logger.debug(
            f"Output frequency modulus: {elapsed.seconds % self.output_freq_sec}"
        )

        return elapsed.seconds % self.output_freq_sec == 0

    def __call__(self, state: FortranState) -> None:
        """
        Hook function for storing the fortran state used by call_py_fort.
        Stores everything that resides in the state at the time.

        'model_time' is expected to be in the state and is removed
        for each storage call.  All other variables are expected to
        correspond to DIMS_MAP after a transpose.

        Args:
            state: Fortran state fields
        """

        state = dict(**state)
        time = _translate_time(state.pop("model_time"))

        if self.initial_time is None:
            self.initial_time = time

        if self._store_interval_check(time):

            logger.debug(
                f"Store flags: save_zarr={self.save_zarr}, save_nc={self.save_nc}"
            )

            if self.save_zarr:
                _store_zarr(state, time, self.monitor, self.metadata)

            if self.save_nc:
                _store_netcdf(state, time, self.nc_dump_path, self.metadata)

            if self.save_tfrecord:
                self._store_tfrecord(state, time)
