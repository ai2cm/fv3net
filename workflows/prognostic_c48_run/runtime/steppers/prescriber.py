from typing import Tuple, Optional, Callable, Mapping
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
        variables: a mapping from variable names in the dataset to the standard names
            to be prescribed
        consolidated: whether desired dataset has consolidated metadata;
            defaults to True
        reference_initial_time: if time interpolating, time of first point in dataset
        reference_frequency_seconds: time frequency of dataset
        apply_interval_seconds: optional- prescribe only on intervals
        tendency_variables: optional mapping of tendency variable names in the dataset
            to the state variables they update

    Example::

        PrescriberConfig(
            dataset_key="gs://vcm-ml-intermediate/2021-03-fine-res-surface-radiative-fluxes/fine-res-surface-radiative-fluxes.zarr",
            variables={
                "DSWRFsfc": "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface",
                "air_temperature": "air_temperature"
            }
        )

    """  # noqa

    dataset_key: str
    variables: Mapping[str, str]
    consolidated: bool = True
    reference_initial_time: Optional[str] = None
    reference_frequency_seconds: float = 900
    apply_interval_seconds: Optional[int] = None
    tendency_variables: Optional[Mapping[str, str]] = None


class Prescriber:

    label = "prescriber"

    def __init__(
        self,
        communicator: pace.util.CubedSphereCommunicator,
        time_lookup_function: Callable[[cftime.DatetimeJulian], State],
        variables: Mapping[str, str],
        tendency_variables: Optional[Mapping[str, str]],
    ):
        """Create a Prescriber object

        Args:
            communicator (pace.util.CubedSphereCommunicator),
            time_lookup_function: a function that takes a time and returns a state dict
                containing data arrays to be prescribed
            variables: a mapping from variable names returned by the
                `time_lookup_function` to the standard names to be prescribed
            tendency_variables: mapping from tendency variable names returned by the
                `time_lookup_function` to the standard tendency names used to update
                the state
        """
        self._communicator = communicator
        self._time_lookup_function = time_lookup_function
        self._variables = variables
        self._tendency_variables = tendency_variables or {}

    def _open_prescribed_timestep(self, time: cftime.DatetimeJulian) -> xr.Dataset:
        ds = scatter_within_tile(time, self._time_lookup_function, self._communicator)
        return ds.rename(**self._variables, **self._tendency_variables)

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}

        prescribed_timestep: xr.Dataset = self._open_prescribed_timestep(time)
        tendency: Tendencies = {}
        state_updates: State = {}

        for name in self._tendency_variables.values():
            tendency[name] = prescribed_timestep[name]

        for name in self._variables.values():
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
        for name, update in tendency.items():
            diagnostics[f"{name}_prescribed_tendency"] = update
        return tendency, diagnostics, state_updates

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        return {}, xr.DataArray()


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
