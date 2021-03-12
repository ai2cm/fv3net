from typing import (
    Union,
    Sequence,
    Callable,
    MutableMapping,
    Mapping,
    Hashable,
)
import dataclasses
from functools import partial
import warnings
import xarray as xr
import cftime
from runtime.steppers.machine_learning import MachineLearningConfig
from runtime.types import State, Diagnostics
import fv3gfs.util
from vcm.catalog import catalog as CATALOG
from vcm.fv3 import metadata
from vcm.convenience import round_time
from vcm.safe import get_variables


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for prescribing states in the model from an external source

    Attributes:
        variables: list variable names to prescribe
        data_source: vcm catalog entry containing variables to prescribe

    Example::

        PrescriberConfig(
            variables=['']
            data_source=""
        )

    """

    variables: Sequence[str]
    data_source: str
    rename: Mapping[str, str]


class Prescriber:
    """A pre-physics stepper which obtains prescribed values from an external source
    
    """

    net_moistening = "net_moistening"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: fv3gfs.util.CubedSphereCommunicator,
    ):

        self._prescribed_variables: Sequence[str] = list(config.variables)
        self._data_source: str = config.data_source
        self._rename: Mapping[str, str] = config.rename
        self._communicator = communicator
        self._load_external_states: Callable[[cftime.DatetimeJulian], State] = partial(
            load_external_states,
            self._data_source,
            self._prescribed_variables,
            self._communicator,
        )

    def __call__(self, time, state):

        diagnostics: Diagnostics = {}
        state_updates: State = DerivedExternalState(
            self._load_external_states(time), self._rename
        )

        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]

        tendency = {}
        return tendency, diagnostics, state_updates

    def get_diagnostics(self, state, tendency):
        return {}

    def get_momentum_diagnostics(self, state, tendency):
        return {}


@dataclasses.dataclass
class PrephysicsConfig:
    """Configuration of pre-physics computations
    
    Attributes:
        config: can be either a MachineLearningConfig or a
            PrescriberConfig, as these are the allowed pre-physics computations
        
    """

    config: Union[PrescriberConfig, MachineLearningConfig]


class DerivedExternalState(Mapping[str, xr.DataArray]):
    def __init__(self, input_state: State, rename: Mapping[str, str]):
        self._state = input_state
        self._rename = rename

    @staticmethod
    def _rename_state(state: State, rename: Mapping[str, str]) -> State:
        new_state: State = {}
        for name in state.keys():
            new_name = rename.get(str(name))
            if new_name:
                new_state[new_name] = state[name]
            else:
                new_state[name] = state[name]
        return new_state

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self._rename_state(self._state, self._rename).keys():
            item = self._rename_state(self._state, self._rename)[key]
        elif key == "total_sky_net_shortwave_flux_at_surface_override":
            item = self._state["DSWRFsfc"] - self._state["USWRFsfc"]
            item = item.assign_attrs(
                {
                    "long_name": "net shortwave radiative flux at surface (downward)",
                    "units": "W/m^2",
                }
            )
        return item

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return set(
            name for name in self._rename_state(self._state, self._rename).keys()
        ) | {"total_sky_net_shortwave_flux_at_surface_override"}


def load_external_states(
    catalog_entry: str,
    variables: Sequence[str],
    communicator: fv3gfs.util.CubedSphereCommunicator,
    time: cftime.DatetimeJulian,
    consolidated: bool = True,
) -> State:
    catalog_ds = _catalog_ds(catalog_entry)
    requested_ds = _requested_ds(catalog_ds, variables, time)
    tile_ds = _tile_ds(requested_ds, communicator)
    state = _quantity_state_to_state(
        communicator.tile.scatter_state(_ds_to_quantity_state(tile_ds))
    )
    return state


def _catalog_ds(catalog_entry: str) -> xr.Dataset:
    try:
        catalog_ds = CATALOG[catalog_entry].to_dask()
    except KeyError:
        raise KeyError(f"Invalid catalog entry provided: {catalog_entry}")
    return catalog_ds


def _requested_ds(
    ds: xr.Dataset, variables: Sequence[str], time: cftime.DatetimeJulian
) -> xr.Dataset:

    ds = (
        ds.pipe(metadata._rename_dims)
        .pipe(metadata._set_missing_attrs)
        .pipe(metadata._remove_name_suffix)
        .pipe(get_variables, variables)
    )
    # this is because vcm.round_time doesn't work in the prognostic_run container
    # which uses python 3.6.9, since singledispatch requires 3.7
    ds = ds.assign_coords({"time": [round_time(time.item()) for time in ds["time"]]})
    try:
        ds = ds.sel(time=time)
    except KeyError:
        raise KeyError(f"Requested time ({time}) not in dataset.")
    return ds


def _tile_ds(
    ds: xr.Dataset, communicator: fv3gfs.util.CubedSphereCommunicator
) -> xr.Dataset:

    if ds.chunks.get("tile", None) != 6:
        warnings.warn(
            "Requested dataset does not have individual tile chunks. This may "
            "be very slow."
        )

    rank = communicator.rank
    tile = communicator.partitioner.tile_index(rank)
    rank_within_tile = communicator.tile.rank

    if rank_within_tile == 0:
        tile_ds = ds.isel(tile=tile).load()

    return tile_ds


def _ds_to_quantity_state(
    state: xr.Dataset,
) -> MutableMapping[Hashable, fv3gfs.util.Quantity]:
    return {
        variable: fv3gfs.util.Quantity.from_data_array(state[variable])
        for variable in state.data_vars
    }


def _quantity_state_to_state(
    quantity_state: MutableMapping[str, fv3gfs.util.Quantity]
) -> State:
    return {
        variable: quantity_state[variable].data_array
        for variable in quantity_state.keys()
    }
