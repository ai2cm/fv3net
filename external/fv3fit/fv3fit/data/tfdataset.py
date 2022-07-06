import dataclasses
from typing import Mapping, Sequence, Optional
import tensorflow as tf
from .base import TFDatasetLoader
from ..tfdataset import iterable_to_tfdataset
import tempfile
import xarray as xr
import numpy as np
from fv3fit._shared.stacking import stack


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
        data = stack(ds[name], unstacked_dims=unstacked_dims).values
        return data


def open_zarr(url: str):
    cachedir = tempfile.mkdtemp()
    return xr.open_zarr(
        "filecache::" + url, storage_options={"filecache": {"cache_storage": cachedir}}
    )


@dataclasses.dataclass
class WindowedZarrLoader(TFDatasetLoader):
    """
    A tfdataset loader that loads directly from zarr and supports time windows.

    Attributes:
        data_path: path to zarr data
        window_size: number of timesteps to include in time windows
        default_variable_config: default configuration for variables
        variable_configs: configuration for variables by name
        batch_size: number of windows to include in each batch
    """

    data_path: str
    unstacked_dims: Sequence[str]
    window_size: int
    default_variable_config: VariableConfig
    variable_configs: Mapping[str, VariableConfig] = dataclasses.field(
        default_factory=dict
    )
    batch_size: int = 1

    def get_data(
        self, local_download_path: Optional[str], variable_names: Sequence[str],
    ) -> tf.data.Dataset:
        """
        Args:
            local_download_path: if provided, cache data locally at this path
            variable_names: names of variables to include when loading data
        Returns:
            dataset containing requested variables
        """
        # using tfdataset.cache(local_download_path)
        ds = open_zarr(self.data_path)

        def records():
            n_times = ds.dims["time"]
            n_windows = n_times // self.window_size
            # must ensure n_windows is evenly divisible by self.batch_size
            n_windows = int((n_windows // self.batch_size) * self.batch_size)
            starts = np.random.randint(0, n_times - self.window_size, n_windows)
            for i_start in starts:
                record = {}
                window_ds = ds.isel(time=range(i_start, i_start + self.window_size))
                for name in variable_names:
                    config = self.variable_configs.get(
                        name, self.default_variable_config
                    )
                    record[name] = config.get_record(
                        name, window_ds, self.unstacked_dims
                    )
                yield record

        tfdataset = iterable_to_tfdataset(records())
        # if local_download_path is given, cache on disk
        if local_download_path is not None:
            tfdataset = tfdataset.cache(local_download_path)
        return tfdataset.unbatch().batch(self.batch_size)
