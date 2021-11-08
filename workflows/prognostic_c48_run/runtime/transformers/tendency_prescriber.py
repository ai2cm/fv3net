import dataclasses
import logging
from typing import Hashable, Mapping, MutableMapping, Set

import cftime
import fsspec
import xarray as xr
import zarr

import fv3gfs.util
from runtime.monitor import Monitor
from runtime.types import Diagnostics, Step
from runtime.derived_state import DerivedFV3State
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TendencyPrescriberConfig:
    """Configuration for overriding tendencies from a step.
    
    Attributes:
        url: path to dataset containing tendencies.
        variables: mapping from state name to name of corresponding tendency in
            provided dataset. For example: {"air_temperature": "fine_res_Q1"}.
    """

    url: str
    variables: Mapping[str, str]


@dataclasses.dataclass
class TendencyPrescriber:
    """Wrap a Step function and prescribe certain tendencies."""

    config: TendencyPrescriberConfig
    state: DerivedFV3State
    communicator: fv3gfs.util.CubedSphereCommunicator
    timestep: float
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "TendencyPrescriber"):
        if self.communicator.rank == 0:
            logger.debug(f"Opening tendency overriding dataset from: {self.config.url}")
        tile = self.communicator.partitioner.tile_index(self.communicator.rank)
        mapper = zarr.LRUStoreCache(fsspec.get_mapper(self.config.url), 128 * 2 ** 20)
        ds = xr.open_zarr(mapper, consolidated=True)
        self._tendency_ds = ds[list(self.config.variables.values())].isel(tile=tile)

    def _open_tendencies_dataset(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        ds = xr.Dataset()
        if self.communicator.tile.rank == 0:
            ds = self._tendency_ds.sel(time=time).drop_vars("time").load()
        tendencies = self.communicator.tile.scatter_state(dataset_to_quantity_state(ds))
        return quantity_state_to_dataset(tendencies)

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    def _prescribe_tendency(self, func: Step) -> Diagnostics:
        tendencies = self._open_tendencies_dataset(self.state.time)
        before = self.monitor.checkpoint()
        diags = func()
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                self.state[variable_name] = (
                    before[variable_name] + tendencies[tendency_name] * self.timestep
                )
        change_due_to_prescribing = self.monitor.compute_change(
            "tendency_prescriber", before, self.state
        )
        return {**diags, **change_due_to_prescribing}

    def __call__(self, func: Step) -> Step:
        """Override tendencies from a function that updates the State.
        
        Args:
            func: a function that updates the State and return Diagnostics.
            
        Returns:
            A function which calls ``func`` and prescribes a given change
            for specified variables.
        """

        def step() -> Diagnostics:
            return self._prescribe_tendency(func)

        step.__name__ = func.__name__
        return step
