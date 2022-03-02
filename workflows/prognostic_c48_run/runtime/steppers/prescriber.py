from typing import Sequence, Tuple, Optional, Callable
import dataclasses
import logging
import cftime
import xarray as xr
from runtime.types import State, Diagnostics, Tendencies
from runtime.scatter import scatter_within_tile
from runtime.names import SST, TSFC, MASK

import pace.util

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for a prescriber object to set states in the model from an external source

    Attributes:
        dataset_key: path of zarr dataset
        variables: sequence of variable names in the dataset to prescribe
        consolidated: whether desired dataset has consolidated metadata;
            defaults to True
        reference_initial_time: if time interpolating, time of first point in dataset
        reference_frequency_seconds: time frequency of dataset

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
    reference_initial_time: Optional[str] = None
    reference_frequency_seconds: float = 900


class Prescriber:

    label = "prescriber"

    def __init__(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        time_lookup_function: Callable[[cftime.DatetimeJulian], State],
    ):
        """Create a Prescriber object

        Args:
            communicator (pace.util.CubedSphereCommunicator),
            time_lookup_function: a function that takes a time and returns a state dict
                containing data arrays to be prescribed
        """
        self._communicator = communicator
        self._time_lookup_function = time_lookup_function

    def _open_prescribed_timestep(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        return scatter_within_tile(time, self._time_lookup_function, self._communicator)

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}

        prescribed_timestep: xr.Dataset = self._open_prescribed_timestep(time)
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
