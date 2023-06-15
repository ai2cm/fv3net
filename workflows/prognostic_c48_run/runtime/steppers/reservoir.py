import dataclasses
from datetime import timedelta

from fv3fit._shared.halos import append_halos_using_mpi
from fv3fit.reservoir.model import HybridReservoirComputingModel


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

    def __init__(self, model: HybridReservoirComputingModel):
        self.model = model
        self.rc_timestep: timedelta
        self.synchronize_steps: int
        self.init_time = None
        self.completed_sync_steps = 0

    def __call__(self, time, state):
        if self.init_time is None:
            self.init_time = time

        # TODO: should I use interval stepper instead?
        if self._is_rc_update_step(time):
            model_inputs = state[self.model.input_variables]
            try:
                n_halo_points = self.model.rank_divider.overlap
                ds_with_halos = append_halos_using_mpi(model_inputs, n_halo_points)
            except RuntimeError:
                raise ValueError(
                    "MPI not available or tile dimension does not exist in state fields"
                    " during RC stepper update"
                )

            if self.completed_sync_steps < self.synchronize_steps:
                # TODO: currently synchronizing takes processed input readout data?
                # TODO: need to make increment handle general data, probably
                self.model.synchronize(ds_with_halos)
                self.completed_sync_steps += 1
                pass
            else:
                updated_state = self.model.predict(ds_with_halos)  # noqa
                # TODO: parse out the state/tendencies if necessary

        return {}, {}, {}

    def _is_rc_update_step(self, time):
        return (time - self.init_time) % self.rc_timestep == 0
