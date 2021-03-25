from typing import Sequence, MutableMapping, Mapping, Hashable, Tuple, Set, Optional
import dataclasses
from datetime import timedelta
import logging
import intake
import cftime
import xarray as xr
import numpy as np
from runtime.types import State, Diagnostics, Tendencies
import fv3gfs.util
from vcm.catalog import catalog as CATALOG
from vcm.convenience import round_time
from vcm.safe import get_variables, warn_if_intersecting


logger = logging.getLogger(__name__)

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]

DIM_RENAME_INVERSE_MAP = {
    "x": {"grid_xt", "grid_xt_coarse"},
    "y": {"grid_yt", "grid_yt_coarse"},
    "tile": {"rank"},
    "x_interface": {"grid_x", "grid_x_coarse"},
    "y_interface": {"grid_y", "grid_y_coarse"},
}


def get_timesteps(
    init_time: cftime.DatetimeJulian, timestep_seconds: float, n_timesteps: int
) -> Sequence[cftime.DatetimeJulian]:
    """Get sequence of model timesteps
    
    Args
        init_time (cftime.DatetimeJulian): model run start time
        timestep_seconds (float): model timestep
        n_timesteps: number of timesteps in model run
    
    Returns: Sequence of cftime.DatetimeJulian objects corresponding to beginnings
        of model run timesteps
    
    """
    return [
        init_time + timedelta(seconds=timestep_seconds * i)
        for i in range(1, n_timesteps + 1)
    ]


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for a prescriber object to set states in the model from an external source

    Attributes:
        dataset_key (str): location of the dataset that provides prescribe values;
            the routine first will try `dataset_key` as a `vcm.catalog` key, and if
            not present it will attempt to use it as a (local or remote) path to
            a zarr dataset
        variables (Sequence[str]): sequence of "standardized" ("_coarse" suffix removed)
            variable names to prescribe
        rename (Mapping[Hashable, Hashable]): mapping of "standardized" ("_coarse"
            suffix removed) names in the external dataset to variable names desired
            for the runfile
        consolidated (bool): optional, whether desired dataset has consolidated metadata;
            defaults to True

    Example::

        PrescriberConfig(
            dataset_key="40day_c48_gfsphysics_15min_may2020"
            variables=["DSWRFsfc", "USWRFsfc", "DLWRFsfc"],
            rename={
                'DSWRFsfc': 'override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface',
                'DLWRFsfc': 'override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface',
                'NSWRFsfc': 'override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface'
            }
        )

    """  # noqa

    dataset_key: str
    variables: Sequence[str]
    rename: Mapping[Hashable, Hashable]
    consolidated: bool = True


class Prescriber:

    net_moistening = "net_moistening"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: fv3gfs.util.CubedSphereCommunicator,
        timesteps: Optional[Sequence[cftime.DatetimeJulian]] = None,
    ):
        """Create a Prescriber object
        
        Args:
            config (PrescriberConfig),
            communicator (fv3gfs.util.CubedSphereCommunicator),
            timesteps (Sequence[cftime.DatetimeJulian]): optional sequence specifying
                all wrapper timesteps for which data is required; if not supplied,
                defaults to downloading entire time dimension of `dataset_key`
        """
        self._config = config
        self._communicator = communicator
        self._rename: Mapping[
            Hashable, Hashable
        ] = config.rename if config.rename else {}
        self._timesteps = timesteps
        self._prescribed_ds: xr.Dataset = self._scatter_prescribed_ds()

    def _scatter_prescribed_ds(self):
        time_coord: Optional[xr.DataArray]
        if self._communicator.rank == 0:
            prescribed_ds, time_coord = _get_prescribed_ds(
                self._config.dataset_key,
                list(self._config.variables),
                self._timesteps,
                self._config.consolidated,
            )
            prescribed_ds = _quantity_state_to_ds(
                self._communicator.scatter_state(_ds_to_quantity_state(prescribed_ds))
            )
        else:
            prescribed_ds = _quantity_state_to_ds(self._communicator.scatter_state())
            time_coord = None
        time_coord = self._communicator.comm.bcast(time_coord, root=0)
        prescribed_ds = prescribed_ds.assign_coords({"time": time_coord})
        return prescribed_ds

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}
        prescribed_timestep: xr.Dataset = _cast_to_double(
            _add_net_shortwave(self._prescribed_ds.sel(time=time))
        )
        state_updates: State = self._rename_state(
            {name: prescribed_timestep[name] for name in prescribed_timestep.data_vars}
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


def _get_prescribed_ds(
    dataset_key: str,
    variables: Sequence[str],
    timesteps: Optional[Sequence[cftime.DatetimeJulian]],
    consolidated: bool = True,
) -> Tuple[xr.Dataset, xr.DataArray]:
    logger.info(f"Setting up dataset for state setting: {dataset_key}")
    ds = _open_ds(dataset_key, consolidated)
    ds = _remove_name_suffix(ds)
    ds = get_variables(ds, variables)
    ds = _rename_dims(ds)
    ds = _set_missing_units_attr(ds)
    ds = _round_time_coord(ds)
    if timesteps is not None:
        ds = ds.sel(time=timesteps)
    time_coord = ds.coords["time"]
    return ds.drop_vars(names="time").load(), time_coord


def _open_ds(dataset_key: str, consolidated: bool) -> xr.Dataset:
    try:
        zarr_path = CATALOG[dataset_key].urlpath
    except KeyError:
        zarr_path = dataset_key
    return intake.open_zarr(zarr_path, consolidated=consolidated).to_dask()


def _rename_dims(
    ds: xr.Dataset, rename_inverse: Mapping[str, Set[str]] = DIM_RENAME_INVERSE_MAP
) -> xr.Dataset:
    varname_target_registry = {}
    for target_name, source_names in rename_inverse.items():
        varname_target_registry.update({name: target_name for name in source_names})
    vars_to_rename = {
        var: varname_target_registry[str(var)]
        for var in ds.dims
        if var in varname_target_registry
    }
    ds = ds.rename(vars_to_rename)
    return ds


def _remove_name_suffix(
    ds: xr.Dataset, suffixes: Sequence[str] = ["_coarse"]
) -> xr.Dataset:
    for target in suffixes:
        replace_names = {
            vname: str(vname).replace(target, "")
            for vname in ds.data_vars
            if target in str(vname)
        }
        warn_if_intersecting(ds.data_vars.keys(), replace_names.values())
        ds = ds.rename(replace_names)
    return ds


def _set_missing_units_attr(ds: xr.Dataset) -> xr.Dataset:
    for var in ds:
        da = ds[var]
        if "units" not in da.attrs:
            da.attrs["units"] = "unspecified"
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


def _round_time_coord(ds: xr.Dataset) -> xr.Dataset:
    # done this way because vcm.round_time's singledispatch doesn't
    # work in the prognostic_run container, which uses python 3.6.9
    time_coord = ds.coords["time"]
    rounded_time_coord = [round_time(time.item()) for time in time_coord]
    return ds.assign_coords({"time": rounded_time_coord})


def _add_net_shortwave(ds: xr.Dataset) -> xr.Dataset:
    net_shortwave = ds["DSWRFsfc"] - ds["USWRFsfc"]
    net_shortwave = net_shortwave.assign_attrs(
        {
            "long_name": "net shortwave radiative flux at surface (downward)",
            "units": "W/m^2",
        }
    )
    ds["NSWRFsfc"] = net_shortwave
    return ds
