from typing import Sequence, Tuple, Optional
import dataclasses
from datetime import timedelta
import logging
import intake
import cftime
import xarray as xr
from runtime.types import State, Diagnostics, Tendencies
from runtime.conversions import quantity_state_to_dataset, dataset_to_quantity_state
from runtime.names import SST, TSFC, MASK

import pace.util
from vcm.catalog import catalog as CATALOG
from vcm.safe import get_variables

logger = logging.getLogger(__name__)

# list of variables that will use nearest neighbor interpolation
# between times instead of linear interpolation
INTERPOLATE_NEAREST = [
    MASK,
]


def get_timesteps(
    init_time: cftime.DatetimeJulian, timestep_seconds: float, n_timesteps: int
) -> Sequence[cftime.DatetimeJulian]:
    """Get sequence of model timesteps
    
    Args
        init_time (cftime.DatetimeJulian): model run start time
        timestep_seconds (float): model timestep
        n_timesteps: number of timesteps in model run
    
    Returns: Sequence of cftime.DatetimeJulian objects corresponding to ends
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
        variables (Sequence[str]): sequence of variable names in the dataset to prescribe
        consolidated (bool): optional, whether desired dataset has consolidated metadata;
            defaults to True

    Example::

        PrescriberConfig(
            dataset_key="gs://vcm-ml-intermediate/2021-03-fine-res-surface-radiative-fluxes/fine-res-surface-radiative-fluxes.zarr",
            variables=[
                "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface", 
                "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface", 
                "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
            ]
        )

    """  # noqa

    dataset_key: str
    variables: Sequence[str]
    consolidated: bool = True


class Prescriber:

    label = "prescriber"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: pace.util.CubedSphereCommunicator,
        timesteps: Optional[Sequence[cftime.DatetimeJulian]] = None,
    ):
        """Create a Prescriber object
        
        Args:
            config (PrescriberConfig),
            communicator (pace.util.CubedSphereCommunicator),
            timesteps (Sequence[cftime.DatetimeJulian]): optional sequence specifying
                all wrapper timesteps for which data is required; if not supplied,
                defaults to downloading entire time dimension of `dataset_key`
        """
        self._config = config
        self._communicator = communicator
        self._timesteps = timesteps
        prescribed_ds, time_coord = self._load_prescribed_ds()
        self._prescribed_ds: xr.Dataset = self._scatter_prescribed_ds(
            prescribed_ds, time_coord
        )

    def _load_prescribed_ds(
        self,
    ) -> Tuple[Optional[xr.Dataset], Optional[xr.DataArray]]:
        prescribed_ds: Optional[xr.Dataset]
        time_coord: Optional[xr.DataArray]
        if self._communicator.rank == 0:
            prescribed_ds, time_coord = _get_prescribed_ds(
                self._config.dataset_key,
                list(self._config.variables),
                self._timesteps,
                self._config.consolidated,
            )
        else:
            prescribed_ds, time_coord = None, None
        return prescribed_ds, time_coord

    def _scatter_prescribed_ds(
        self, prescribed_ds: Optional[xr.Dataset], time_coord: Optional[xr.DataArray]
    ) -> xr.Dataset:
        if isinstance(prescribed_ds, xr.Dataset):
            scattered_ds = quantity_state_to_dataset(
                self._communicator.scatter_state(
                    dataset_to_quantity_state(prescribed_ds)
                )
            )
        else:
            scattered_ds = quantity_state_to_dataset(self._communicator.scatter_state())
        time_coord = self._communicator.comm.bcast(time_coord, root=0)
        scattered_ds = scattered_ds.assign_coords({"time": time_coord})
        return scattered_ds

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}
        prescribed_timestep: xr.Dataset = self._prescribed_ds.sel(time=time)
        state_updates: State = {}

        for name in prescribed_timestep.data_vars:
            if name == MASK:
                state_updates[name] = prescribed_timestep[name].round()

            elif name == SST:
                # If just the sea surface temperature is to be updated
                # (and not land as well), the prescribed dataarray should be
                # "ocean_surface_temperature".
                state_updates.update(
                    sst_update_from_reference(state, prescribed_timestep, SST)
                )

            else:
                state_updates[name] = prescribed_timestep[name]
        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]
        tendency: Tendencies = {}
        return tendency, diagnostics, state_updates

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        return {}, xr.DataArray()

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        return {}


def _get_prescribed_ds(
    dataset_key: str,
    variables: Sequence[str],
    timesteps: Optional[Sequence[cftime.DatetimeJulian]],
    consolidated: bool = True,
) -> Tuple[xr.Dataset, xr.DataArray]:
    logger.info(f"Setting up dataset for state setting: {dataset_key}")
    ds = _open_ds(dataset_key, consolidated)
    ds = get_variables(ds, variables)
    if timesteps is not None:
        ds = _time_interpolate_data(ds, timesteps, variables)
    time_coord = ds.coords["time"]
    return ds.drop_vars(names="time").load(), time_coord


def _time_interpolate_data(ds, timesteps, variables):
    vars_interp_nearest = [var for var in variables if var in INTERPOLATE_NEAREST]
    vars_interp_linear = [var for var in variables if var not in INTERPOLATE_NEAREST]

    ds_interp = xr.Dataset()
    if len(vars_interp_nearest) > 0:
        ds_interp_nearest = ds[vars_interp_nearest].interp(
            time=timesteps, method="nearest"
        )
        ds_interp.update(ds_interp_nearest)
    if len(vars_interp_linear) > 0:
        ds_interp_linear = ds[vars_interp_linear].interp(time=timesteps)
        ds_interp.update(ds_interp_linear)
    return ds_interp


def _open_ds(dataset_key: str, consolidated: bool) -> xr.Dataset:
    try:
        ds = CATALOG[dataset_key].to_dask()
    except KeyError:
        ds = intake.open_zarr(dataset_key, consolidated=consolidated).to_dask()
    return ds


def _sst_from_reference(
    reference_surface_temperature: xr.DataArray,
    surface_temperature: xr.DataArray,
    land_sea_mask: xr.DataArray,
) -> xr.DataArray:
    # prescribes SST but does not update state surface temperature over land
    return xr.where(
        land_sea_mask.values.round().astype("int") == 0,
        reference_surface_temperature,
        surface_temperature,
    ).assign_attrs(units=surface_temperature.units)


def sst_update_from_reference(
    state: State, reference: State, reference_sst_name=TSFC
) -> State:
    """
    Set the sea surface and surface temperatures in a model state to values in
    a reference state. Useful for maintaining consistency between a nudged run
    and reference state.
    """
    slmsk = reference[MASK].round() if MASK in reference else state[MASK]
    state_updates: State = {
        SST: _sst_from_reference(reference[reference_sst_name], state[SST], slmsk),
        TSFC: _sst_from_reference(reference[reference_sst_name], state[TSFC], slmsk),
    }
    return state_updates
