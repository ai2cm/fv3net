import dataclasses
from typing import Hashable, Mapping, MutableMapping, Optional, Set, Tuple

import fsspec
import xarray as xr

import fv3gfs.util
from runtime.monitor import Monitor
from runtime.types import State, Diagnostics, Step
from runtime.derived_state import DerivedFV3State

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]


@dataclasses.dataclass
class OverriderConfig:
    """Configuration for overriding tendencies from a step.
    
    Attributes:
        url: path to dataset containing tendencies.
        variables: mapping from state name to name of corresponding tendency in
            provided dataset. For example: {"air_temperature": "fine_res_Q1"}.
    """

    url: str
    variables: Mapping[str, str]


@dataclasses.dataclass
class OverriderAdapter:
    """Wrap a Step function to override certain tendencies."""

    config: OverriderConfig
    state: DerivedFV3State
    communicator: fv3gfs.util.CubedSphereCommunicator
    timestep: float
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "OverriderAdapter"):
        ds, time_coord = self._open_tendencies_dataset()
        self.tendencies = self._scatter_dataset(ds, time_coord)

    def _open_tendencies_dataset(
        self,
    ) -> Tuple[Optional[xr.Dataset], Optional[xr.DataArray]]:
        if self.communicator.rank == 0:
            mapper = fsspec.get_mapper(self.config.url)
            ds = xr.open_zarr(mapper, consolidated=True)
            ds = ds[list(self.config.variables.values())]
            time_coord = ds.coords["time"]
            ds = ds.drop_vars("time")
        else:
            ds = None
            time_coord = None
        return ds, time_coord

    def _scatter_dataset(
        self, ds: Optional[xr.Dataset], time_coord: Optional[xr.DataArray]
    ) -> xr.Dataset:
        if ds is not None:
            state = self.communicator.scatter_state(_ds_to_quantity_state(ds))
        else:
            state = self.communicator.scatter_state()
        time_coord = self.communicator.comm.bcast(time_coord, root=0)
        return _quantity_state_to_ds(state).assign_coords({"time": time_coord})

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    def override(self, name: str, func: Step) -> Diagnostics:
        tendencies = self.tendencies.sel(time=self.state.time).load()
        before = self.monitor.checkpoint()
        diags = func()
        change_due_to_func = self.monitor.compute_change(name, before, self.state)
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                self.state[variable_name] = (
                    before[variable_name] + tendencies[tendency_name] * self.timestep
                )
        change_due_to_overriding = self.monitor.compute_change(
            "override", before, self.state
        )
        return {**diags, **change_due_to_func, **change_due_to_overriding}

    def __call__(self, name: str, func: Step) -> Step:
        """Override tendencies from a function that updates the State.
        
        Args:
            name: a label for the step that is being overriden.
            func: a function that updates the State and return Diagnostics.
            
        Returns:
            overridden_func: a function which observes the change to State
            done by ``func`` and overrides the change for specified variables.
        """

        def step() -> Diagnostics:
            return self.override(name, func)

        step.__name__ = func.__name__
        return step


def _ds_to_quantity_state(ds: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: fv3gfs.util.Quantity.from_data_array(ds[variable])
        for variable in ds.data_vars
    }
    return quantity_state


def _quantity_state_to_ds(quantity_state: QuantityState) -> xr.Dataset:
    ds = xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
    return ds
