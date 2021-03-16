from typing import (
    Sequence,
    MutableMapping,
    Mapping,
    Hashable,
)
import dataclasses
import warnings
import xarray as xr
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
    catalog_entry: str
    rename: Mapping[str, str]


class Prescriber:
    """A pre-physics stepper which obtains prescribed values from an external source
    
    """

    net_moistening = "net_moistening"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: fv3gfs.util.CubedSphereCommunicator,
        rename: Mapping[str, str],
    ):

        self._rename = config.rename
        self._dataset: xr.Dataset = setup_dataset(
            config.catalog_entry, list(config.variables), communicator
        )

    def __call__(self, time, state):

        diagnostics: Diagnostics = {}
        state_updates: State = DerivedExternalDataset(
            self._dataset.sel(time=time), self._rename
        )

        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]

        tendency = {}
        return tendency, diagnostics, state_updates

    def get_diagnostics(self, state, tendency):
        return {}

    def get_momentum_diagnostics(self, state, tendency):
        return {}


class DerivedExternalDataset(Mapping[str, xr.DataArray]):
    def __init__(self, input_ds: xr.Dataset, rename: Mapping[str, str]):
        self._dataset: xr.Dataset = self._rename_ds(input_ds, rename)

    def _rename_ds(self, ds: xr.Dataset, rename: Mapping[str, str]) -> xr.Dataset:
        new_rename = {}
        for name in ds.data_vars:
            if name in rename.keys():
                new_rename[name] = rename[name]
        return ds.rename(new_rename)

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self._dataset.keys():
            item = self._dataset[key]
        elif key == "total_sky_net_shortwave_flux_at_surface_override":
            item = self._dataset["DSWRFsfc"] - self._dataset["USWRFsfc"]
            item = item.assign_attrs(
                {
                    "long_name": "net shortwave radiative flux at surface (downward)",
                    "units": "W/m^2",
                }
            )
        else:
            raise KeyError(f"Requested value not in derived external dataset: {key}")
        return item

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self.keys())

    def keys(self):
        return set(name for name in self._dataset.keys()) | {
            "total_sky_net_shortwave_flux_at_surface_override"
        }


def setup_dataset(
    catalog_entry: str,
    variables: Sequence[str],
    communicator: fv3gfs.util.CubedSphereCommunicator
) -> xr.Dataset:
    catalog_ds = _catalog_ds(catalog_entry)
    requested_ds = _requested_ds(catalog_ds, variables)
    tile_ds = _tile_ds(requested_ds, communicator)
    ds = _quantity_state_to_ds(
        communicator.tile.scatter_state(_ds_to_quantity_state(tile_ds))
    )
    return ds


def _catalog_ds(catalog_entry: str) -> xr.Dataset:
    try:
        catalog_ds = CATALOG[catalog_entry].to_dask()
    except KeyError:
        raise KeyError(f"Invalid catalog entry provided: {catalog_entry}")
    return catalog_ds


def _requested_ds(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:

    ds = (
        ds.pipe(metadata._rename_dims)
        .pipe(metadata._set_missing_attrs)
        .pipe(metadata._remove_name_suffix)
        .pipe(get_variables, variables)
    )
    # this is because vcm.round_time doesn't work in the prognostic_run container
    # which uses python 3.6.9, since singledispatch requires 3.7
    ds = ds.assign_coords({"time": [round_time(time.item()) for time in ds["time"]]})
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


def _quantity_state_to_ds(
    quantity_state: MutableMapping[str, fv3gfs.util.Quantity]
) -> xr.Dataset:
    return xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
