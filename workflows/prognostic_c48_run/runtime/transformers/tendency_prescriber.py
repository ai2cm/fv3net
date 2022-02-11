import dataclasses
import logging
from datetime import timedelta
from typing import Mapping, Set, Optional

import cftime
import xarray as xr

import pace.util
import loaders
import vcm
from runtime.monitor import Monitor
from runtime.types import Diagnostics, Step, State
from runtime.derived_state import DerivedFV3State
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state
from runtime.interpolate import time_interpolate_func, label_to_time

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TendencyPrescriberConfig:
    """Configuration for overriding tendencies from a step.
    
    Attributes:
        mapper_config: configuration of mapper used to load tendency data.
        variables: mapping from state name to name of corresponding tendency in
            provided mapper. For example: {"air_temperature": "fine_res_Q1"}.
    """

    mapper_config: loaders.MapperConfig
    variables: Mapping[str, str]
    reference_initial_time: Optional[str] = None
    reference_frequency_seconds: float = 900


@dataclasses.dataclass
class TendencyPrescriber:
    """Wrap a Step function and prescribe certain tendencies."""

    config: TendencyPrescriberConfig
    state: DerivedFV3State
    communicator: pace.util.CubedSphereCommunicator
    timestep: float
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "TendencyPrescriber"):
        if self.communicator.rank == 0:
            logger.debug(f"Opening tendency override from: {self.config.mapper_config}")
        mapper = self.config.mapper_config.load_mapper()
        initial_label = self.config.reference_initial_time

        def mapper_func(time: cftime.DatetimeJulian) -> State:
            timestamp = vcm.encode_time(time)
            ds = mapper[timestamp]
            return {var: ds[var] for var in ds.data_vars}

        if initial_label is not None:
            self._mapper_func = time_interpolate_func(
                mapper_func,
                frequency=timedelta(seconds=self.config.reference_frequency_seconds),
                initial_time=label_to_time(initial_label),
            )
        else:
            self._mapper_func = mapper_func
        self._tendency_names = list(self.config.variables.values())

    def _open_tendencies_dataset(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        tile = self.communicator.partitioner.tile_index(self.communicator.rank)
        if self.communicator.tile.rank == 0:
            ds = (
                xr.Dataset(self._mapper_func(time))
                .isel(tile=tile)[self._tendency_names]
                .load()
            )
        else:
            ds = xr.Dataset()
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
