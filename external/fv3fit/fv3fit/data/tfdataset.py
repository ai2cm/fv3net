import dataclasses
import os
import cftime
from typing import List, Mapping, Sequence, Optional
import tensorflow as tf
from .base import TFDatasetLoader, register_tfdataset_loader, tfdataset_loader_from_dict
import dacite
from ..tfdataset import generator_to_tfdataset
import tempfile
import xarray as xr
import numpy as np


SAMPLE_DIM_NAME = "_fv3fit_sample"


def stack(ds: xr.DataArray, unstacked_dims: Sequence[str]):
    stack_dims = [dim for dim in ds.dims if dim not in unstacked_dims]
    unstacked_dims = [dim for dim in unstacked_dims if dim in ds.dims]
    if len(stack_dims) == 0:
        ds_stacked = ds.expand_dims(dim=SAMPLE_DIM_NAME, axis=0)
    else:
        ds_stacked = ds.stack({SAMPLE_DIM_NAME: stack_dims})
    return ds_stacked.transpose(SAMPLE_DIM_NAME, *unstacked_dims)


@dataclasses.dataclass
class VariableConfig:
    """
    Configuration for a variable to retrieve from a dataset.

    Attributes:
        times: time indices to use, must be one of "window" or "start", which
            will use either the full window or just the initial time of that window
    """

    times: str = "window"

    def __post_init__(self):
        if self.times not in ("window", "start"):
            raise TypeError("times must be one of 'window' or 'start'")

    def get_record(self, name: str, ds: xr.Dataset, unstacked_dims: Sequence[str]):
        if self.times == "start":
            ds = ds.isel(time=0)
        array = stack(ds[name], unstacked_dims)
        return array.values


def open_zarr_using_filecache(url: str, decode_times: bool = False):
    cachedir = tempfile.mkdtemp()
    return xr.open_zarr(
        "filecache::" + url,
        storage_options={"filecache": {"cache_storage": cachedir}},
        decode_times=decode_times,
    )


def get_n_windows(n_times: int, window_size: int) -> int:
    """
    Args:
        n_times: number of time snapshots in the dataset
        window_size: number of timesteps to include in time windows, note that
            to account for n_steps timesteps, you must have a window_size of
            n_steps + 1 to include both the start and end point of the window

    Returns:
        number of time windows required to fully cover the dataset, given overlapping
            start and end points of each window
    """
    # this function is hard to derive we've included plenty of unit tests of it
    return (n_times - 1) // (window_size - 1)


@register_tfdataset_loader
@dataclasses.dataclass
class CycleGANLoader(TFDatasetLoader):

    domain_configs: List[TFDatasetLoader] = dataclasses.field(default_factory=list)
    batch_size: int = 1

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        datasets = []
        for config in self.domain_configs:
            datasets.append(
                config.open_tfdataset(local_download_path, variable_names).unbatch()
            )
        return tf.data.Dataset.zip(tuple(datasets)).batch(batch_size=self.batch_size)

    @classmethod
    def from_dict(cls, d: dict) -> "CycleGANLoader":
        domain_configs = [
            tfdataset_loader_from_dict(domain_config)
            for domain_config in d["domain_configs"]
        ]
        kwargs = d.copy()
        kwargs["domain_configs"] = domain_configs
        return CycleGANLoader(**kwargs)


@register_tfdataset_loader
@dataclasses.dataclass
class WindowedZarrLoader(TFDatasetLoader):
    """
    A tfdataset loader that loads directly from zarr and supports time windows.

    Windows starts are selected randomly with replacement, so every time will not
    necessarily appear in each iteration over the tf.data.Dataset. Each sample window
    is independently selected along any stacked (sample) dimensions.

    If the "time" variable itself is loaded, it will be converted to a number of
    seconds since 1970-01-01. It is also required that the time variable be
    one-dimensional in "time" (as opposed to varying along some perturbation axis).

    Attributes:
        data_path: path to zarr data
        unstacked_dims: dimensions to keep unstacked when loading data, data loaded
            will have dimensionality [sample] + unstacked_dims
        window_size: number of timesteps to include in time windows, note that
            to account for n_steps timesteps, you must have a window_size of
            n_steps + 1 to include both the start and end point of the window
        default_variable_config: default configuration for variables
        variable_configs: configuration for variables by name
        batch_size: number of samples to include in each batch
        time_stride: use every time_stride time step in windows, useful for
            increasing the model timestep. Number of samples in each window
            is unchanged.
        n_windows: number of windows to create per epoch, defaults to the number
            of windows needed to fully cover the dataset with overlapping
            start and end times.
        time_start_index: index of the first time step of the raw dataset to use,
            defaults to 0
        time_end_index: index of the last time step of the raw dataset to use,
            defaults to the last time step in the dataset
    """

    data_path: str
    unstacked_dims: Sequence[str]
    window_size: int
    default_variable_config: VariableConfig
    variable_configs: Mapping[str, VariableConfig] = dataclasses.field(
        default_factory=dict
    )
    batch_size: int = 1
    time_stride: int = 1
    n_windows: Optional[int] = None
    time_start_index: int = 0
    time_end_index: Optional[int] = None

    def open_tfdataset(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        if "time" in variable_names:
            decode_times = True
        else:
            # if time is not requested, we can skip decoding it
            # as decoding can cause errors if time is poorly formatted in the dataset
            decode_times = False
        ds = open_zarr_using_filecache(self.data_path, decode_times=decode_times)
        ds = ds.isel(time=slice(self.time_start_index, self.time_end_index))
        tfdataset = self._convert_to_tfdataset(ds, variable_names)
        # if local_download_path is given, cache on disk
        if local_download_path is not None:
            os.makedirs(local_download_path, exist_ok=True)
            tfdataset = tfdataset.cache(local_download_path)
        return tfdataset.batch(self.batch_size)

    def _convert_to_tfdataset(
        self, ds: xr.Dataset, variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            ds: xarray data to convert to tfdataset
            variable_names: names of variables to include when loading data
        Returns:
            tfdataset containing requested variables, each record is a mapping from
                variable name to variable value, and each value is a tensor whose
                first dimension is the batch dimension
        """
        tfdataset = generator_to_tfdataset(
            records(
                n_windows=self.n_windows,
                window_size=self.window_size,
                time_stride=self.time_stride,
                ds=ds,
                variable_names=variable_names,
                default_variable_config=self.default_variable_config,
                variable_configs=self.variable_configs,
                unstacked_dims=self.unstacked_dims,
            )
        )
        return tfdataset

    @classmethod
    def from_dict(cls, d: dict) -> "WindowedZarrLoader":
        return dacite.from_dict(
            data_class=cls, data=d, config=dacite.Config(strict=True)
        )


def records(
    n_windows: Optional[int],
    window_size: int,
    time_stride: int,
    ds: xr.Dataset,
    variable_names: Sequence[str],
    default_variable_config: VariableConfig,
    variable_configs: Mapping[str, VariableConfig],
    unstacked_dims: Sequence[str],
):
    time_stride = int(time_stride)

    def generator():
        nonlocal n_windows
        n_times = ds.dims["time"]
        if n_windows is None:
            n_windows = get_n_windows(n_times, window_size * time_stride)
        starts = np.random.randint(0, n_times - window_size * time_stride, n_windows)
        for i_start in starts:
            record = {}
            window_ds = ds.isel(
                time=range(i_start, i_start + window_size * time_stride, time_stride)
            )
            # need to select same random sample for all variable names, but don't know
            # how many samples there are until we look at the first variable
            i_sample = None
            for name in variable_names:
                config = variable_configs.get(name, default_variable_config)
                array = config.get_record(name, window_ds, unstacked_dims)
                if (
                    array.shape[0] == 1
                ):  # special handling for values constant across samples
                    sample = array[0, :]
                else:
                    if i_sample is None:
                        i_sample = np.random.randint(array.shape[0])
                    if len(array.shape) > 1:
                        sample = array[i_sample, :]
                    else:
                        sample = array[i_sample]
                if name == "time":
                    try:
                        item = cftime.date2num(sample, "seconds since 1970-01-01")
                    except ValueError:  # raised for arrays of datetime64
                        item = (
                            sample - np.datetime64("1970-01-01T00:00:00Z")
                        ) / np.timedelta64(1, "s")
                else:
                    try:
                        item = sample
                    except IndexError:
                        item = np.asarray(sample)
                record[name] = item
            yield record

    return generator
