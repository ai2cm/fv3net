import dataclasses
import os
from typing import List, Mapping, Sequence, Optional
import tensorflow as tf
from .base import TFDatasetLoader, register_tfdataset_loader, tfdataset_loader_from_dict
import dacite
from ..tfdataset import generator_to_tfdataset
import tempfile
import xarray as xr
import numpy as np


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
        for dim in unstacked_dims[:-1]:
            if dim not in ds[name].dims:
                raise ValueError("variable {} has no dimension {}".format(name, dim))
        if self.times == "start":
            ds = ds.isel(time=0)
        dims = [d for d in unstacked_dims if d in ds[name].dims]
        data = ds[name].transpose(*dims).values
        return data


def open_zarr_using_filecache(url: str):
    cachedir = tempfile.mkdtemp()
    return xr.open_zarr(
        "filecache::" + url,
        storage_options={"filecache": {"cache_storage": cachedir}},
        decode_times=False,
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
    necessarily appear in each iteration over the tf.data.Dataset.

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
        ds = open_zarr_using_filecache(self.data_path)
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
            for name in variable_names:
                config = variable_configs.get(name, default_variable_config)
                record[name] = config.get_record(name, window_ds, unstacked_dims)
            yield record

    return generator
