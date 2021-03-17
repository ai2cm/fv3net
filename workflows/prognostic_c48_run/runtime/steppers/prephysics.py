from typing import Sequence, MutableMapping, Mapping, Hashable, Tuple
import dataclasses
import warnings
import logging
import intake
import xarray as xr
import numpy as np
from runtime.types import State, Diagnostics, Tendencies
import fv3gfs.util
from vcm.catalog import catalog as CATALOG
from vcm.fv3 import metadata
from vcm.convenience import round_time
from vcm.safe import get_variables


logger = logging.getLogger(__name__)

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for prescribing states in the model from an external source

    Attributes:
        catalog_entry: vcm catalog key of catalog dataset containing variables
        variables: list variable names to prescribe

    Example::

        PrescriberConfig(
            variables=["DSWRFsfc", "USWRFsfc", "DLWRFsfc"]
            catalog_entry="40day_c48_gfsphysics_15min_may2020"
        )

    """

    catalog_entry: str
    variables: Sequence[str]
    rename: Mapping[Hashable, Hashable]
    consolidated: bool = True


class Prescriber:
    """A pre-physics stepper which obtains prescribed values from an external source
    
    """

    net_moistening = "net_moistening"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: fv3gfs.util.CubedSphereCommunicator,
    ):

        self._rename: Mapping[
            Hashable, Hashable
        ] = config.rename if config.rename else {}
        self._dataset: xr.Dataset = setup_dataset(
            config.catalog_entry,
            list(config.variables),
            communicator,
            config.consolidated,
        )

    def __call__(self, time, state):

        diagnostics: Diagnostics = {}
        prescribed_ds = DerivedExternalMapping(self._dataset.sel(time=time))
        state_updates: State = self._rename_state(
            {name: prescribed_ds[name] for name in prescribed_ds.keys()}
        )

        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]

        tendency: Tendencies = {}
        return tendency, diagnostics, state_updates

    def _rename_state(self, state: State) -> State:
        new_state: State = {}
        for name in state.keys():
            if name in self._rename.keys():
                new_state[self._rename[name]] = state[name]
            else:
                new_state[name] = state[name]
        return new_state

    def get_diagnostics(self, state, tendency):
        return {}

    def get_momentum_diagnostics(self, state, tendency):
        return {}


class DerivedExternalMapping(Mapping[str, xr.DataArray]):
    def __init__(self, input_ds: xr.Dataset):
        self._dataset: xr.Dataset = input_ds

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        if key in self._dataset.keys():
            item = self._dataset[key]
        elif key == "NSWRFsfc":
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
        return (set(name for name in self._dataset.keys()) | {"NSWRFsfc"}) - {
            "USWRFsfc"
        }


def setup_dataset(
    catalog_entry: str,
    variables: Sequence[str],
    communicator: fv3gfs.util.CubedSphereCommunicator,
    consolidated: bool = True,
) -> xr.Dataset:
    logger.info(f"Setting up catalog dataset for state setting: {catalog_entry}")
    catalog_ds, time_coord = _catalog_ds(catalog_entry, consolidated)
    # has to be done this way because vcm.round_time's singledispatch doesn't
    # work in the prognostic_run container, which uses python 3.6.9
    rounded_time_coord = [round_time(time.item()) for time in time_coord]
    requested_ds = _requested_ds(catalog_ds, variables)
    tile_ds = _tile_ds(requested_ds, communicator)
    ds = _quantity_state_to_ds(
        communicator.tile.scatter_state(_ds_to_quantity_state(tile_ds))
    )
    ds = _cast_to_double(ds)
    return ds.assign_coords({"time": rounded_time_coord})


def _catalog_ds(
    catalog_entry: str, consolidated: bool
) -> Tuple[xr.Dataset, xr.DataArray]:
    try:
        catalog_entry_path = CATALOG[catalog_entry].urlpath
    except KeyError:
        raise KeyError(f"Invalid catalog entry provided: {catalog_entry}")
    catalog_ds = intake.open_zarr(
        catalog_entry_path, consolidated=consolidated
    ).to_dask()
    time_coord = catalog_ds.coords["time"]
    return catalog_ds.drop_vars("time"), time_coord


def _requested_ds(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    ds = (
        ds.pipe(metadata._remove_name_suffix)
        .pipe(get_variables, variables)
        .pipe(metadata._rename_dims)
        .pipe(metadata._set_missing_attrs)
    )
    return ds


def _cast_to_double(ds: xr.Dataset) -> xr.Dataset:
    new_ds = {}
    for name in ds.data_vars:
        if ds[name].values.dtype != np.float64:
            new_ds[name] = (
                ds[name]
                .astype(np.float64, casting="same_kind")
                .assign_attrs(ds[name].attrs)
            )
        else:
            new_ds[name] = ds[name]
    return xr.Dataset(new_ds).assign_attrs(ds.attrs)


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


def _ds_to_quantity_state(state: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: fv3gfs.util.Quantity.from_data_array(state[variable])
        for variable in state.data_vars
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
