import functools

import fv3gfs.util
import fv3gfs.wrapper
from runtime.nudging import (
    NudgingConfig,
    get_nudging_tendency,
    get_reference_surface_temperatures,
    nudging_timescales_from_dict,
    setup_get_reference_state,
)
from runtime.diagnostics import compute_diagnostics
from runtime.names import SST, TSFC


class PureNudger:

    label = "nudging"

    def __init__(
        self, config: NudgingConfig, communicator: fv3gfs.util.CubedSphereCommunicator,
    ):

        variables_to_nudge = list(config.timescale_hours)
        self._get_reference_state = setup_get_reference_state(
            config,
            variables_to_nudge + [SST, TSFC],
            fv3gfs.wrapper.get_tracer_metadata(),
            communicator,
        )

        self._nudging_timescales = nudging_timescales_from_dict(config.timescale_hours)
        self._get_nudging_tendency = functools.partial(
            get_nudging_tendency, nudging_timescales=self._nudging_timescales,
        )

    def __call__(self, time, state):
        reference = self._get_reference_state(time)
        tendencies = get_nudging_tendency(state, reference, self._nudging_timescales)
        ssts = get_reference_surface_temperatures(state, reference)

        reference = {
            f"{key}_reference": reference_state
            for key, reference_state in reference.items()
        }
        return tendencies, reference, ssts

    def get_diagnostics(self, state, tendency):
        diags = compute_diagnostics(state, tendency, self.label)
        return diags, diags[f"net_moistening_due_to_{self.label}"]

    def get_momentum_diagnostics(self, state, tendency):
        return {}
