import cftime
import dataclasses
from datetime import timedelta
from typing import Optional, MutableMapping, Hashable
import xarray as xr

from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.model import HybridReservoirComputingModel, HybridDatasetAdapter


@dataclasses.dataclass
class HybridReservoirConfig:
    """
    Hybrid reservoir configuration.

    Attributes:
        model: URL to the global hybrid RC model
    """

    model: str
    synchronize_steps: int = 1  # TODO: this could also be set as a duration
    reservoir_timestep: str = "3h"  # TODO: could this be inferred from the model?


class ReservoirStepper:

    label = "hybrid_reservoir"

    def __init__(
        self,
        model: HybridDatasetAdapter,
        reservoir_timestep: timedelta,
        synchronize_steps: int,
        model_timestep_seconds: int = 900,
    ):
        self.model = model
        self.rc_timestep = reservoir_timestep
        self.dt_atmos = timedelta(seconds=model_timestep_seconds)
        self.synchronize_steps = synchronize_steps
        self.init_time: Optional[cftime.DatetimeJulian] = None
        self.completed_sync_steps = 0

    def __call__(self, time, state):
        if self.init_time is None:
            self.init_time = time - self.dt_atmos

        if self._is_rc_update_step(time):
            reservoir_inputs = state[self.model.input_variables]
            hybrid_inputs = state[self.model.hybrid_variables]
            # TODO: I think there needs to be state before the underlying model updates
            # TODO: for the inputs to be correct timewise?

            try:
                n_halo_points = self.model.rank_divider.overlap
                rc_in_with_halos = append_halos_using_mpi(
                    reservoir_inputs, n_halo_points
                )
                hybrid_in_with_halos = append_halos_using_mpi(
                    hybrid_inputs, n_halo_points
                )
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state fields"
                    " during RC stepper update"
                )

            if self.completed_sync_steps == 0:
                self.model.reset_state()

            self.model.increment_state(rc_in_with_halos)
            self.completed_sync_steps += 1
            if self.completed_sync_steps >= self.synchronize_steps:
                updated_state = self.model.predict(hybrid_in_with_halos)

        return {}, {}, updated_state

    def _is_rc_update_step(self, time):
        return (time - self.init_time) % self.rc_timestep == 0

    def get_diagnostics(self, state, tendency):
        diags: MutableMapping[Hashable, xr.DataArray] = {}
        return diags, xr.DataArray()


def open_rc_model(config: HybridReservoirConfig):
    return HybridDatasetAdapter(HybridReservoirComputingModel.load(config.model))
