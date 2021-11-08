from collections import Counter
from typing import List, Union, MutableMapping, Hashable, Tuple
import xarray as xr

from runtime.steppers.machine_learning import PureMLStepper
from runtime.steppers.prescriber import Prescriber
from runtime.types import Diagnostics


def _merge_outputs(
    outputs: List[MutableMapping[Hashable, xr.DataArray]]
) -> MutableMapping[Hashable, xr.DataArray]:
    return {k: v for d in outputs for k, v in d.items()}


def _check_for_collisions(outputs: List[MutableMapping[Hashable, xr.DataArray]]):
    output_keys = [list(output) for output in outputs]
    all_keys: List = sum(output_keys, [])
    key_counts = Counter(all_keys)
    collisions = []
    for key, count in key_counts.items():
        if count > 1:
            collisions.append(key)
    if len(collisions) > 0:
        raise ValueError(f"Outputs have overlapping update keys: {collisions}")


class CombinedStepper:
    def __init__(self, steppers: List[Union[Prescriber, PureMLStepper]]):
        if len(steppers) == 0:
            raise ValueError("No steppers provided to combine.")
        self._steppers = steppers
        # have to call the steppers first before checking if they collide
        self._verified_no_collisions = False

    def __call__(self, time, state):
        tendencies, diagnostics, state_updates = [], [], []
        for stepper in self._steppers:
            t, d, s = stepper(time, state)
            tendencies.append(t)
            diagnostics.append(d)
            state_updates.append(s)
        if self._verified_no_collisions is False:
            for outputs in [tendencies, diagnostics, state_updates]:
                _check_for_collisions(outputs)
                self._verified_no_collisions = True
        return (
            _merge_outputs(tendencies),
            _merge_outputs(diagnostics),
            _merge_outputs(state_updates),
        )

    def get_diagnostics(self, state, tendency) -> Tuple[Diagnostics, xr.DataArray]:
        """Return diagnostics mapping and net moistening array."""
        diags, net_moistening = [], []
        for stepper in self._steppers:
            stepper_diags, stepper_net_moistening = stepper.get_diagnostics(
                state, tendency
            )
            diags.append(stepper_diags)
            if len(stepper_net_moistening.sizes) > 0:
                net_moistening.append(stepper_net_moistening)
        if len(net_moistening) == 0:
            moistening_diag = xr.DataArray()
        elif len(net_moistening) == 1:
            moistening_diag = net_moistening[0]
        else:
            raise ValueError(
                "More than one stepper outputs a net moistening diagnostic. "
                "Only one stepper should be providing this diag."
            )
        _check_for_collisions(diags)
        return _merge_outputs(diags), moistening_diag

    def get_momentum_diagnostics(self, state, tendency) -> Diagnostics:
        """Return diagnostics of momentum tendencies."""
        momentum_diags = []
        for stepper in self._steppers:
            momentum_diags.append(stepper.get_momentum_diagnostics(state, tendency))
        _check_for_collisions(momentum_diags)
        return _merge_outputs(momentum_diags)
