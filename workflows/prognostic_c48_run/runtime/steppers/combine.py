from collections import Counter
from typing import Optional, List, Union, MutableMapping, Hashable
import xarray as xr

from runtime.steppers.machine_learning import PureMLStepper
from runtime.steppers.prescriber import Prescriber


def _merge_outputs(
    outputs: List[MutableMapping[Hashable, xr.DataArray]]
) -> MutableMapping[Hashable, xr.DataArray]:
    return {k: v for d in outputs for k, v in d.items()}


class CombinedStepper:
    def __init__(self, steppers: Optional[List[Union[Prescriber, PureMLStepper]]]):
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
                self._check_for_collisions(outputs)
        return (
            _merge_outputs(tendencies),
            _merge_outputs(diagnostics),
            _merge_outputs(state_updates),
        )

    def _check_for_collisions(
        self, outputs: List[MutableMapping[Hashable, xr.DataArray]]
    ):
        output_keys = [list(output) for output in outputs]
        all_keys: List = sum(output_keys, [])
        key_counts = Counter(all_keys)
        collisions = []
        for key, count in key_counts.items():
            if count > 1:
                collisions.append(key)
        if len(collisions) > 0:
            raise ValueError(
                "Steppers configured in prephysics state updates have overlapping "
                f"update keys: {collisions}"
            )
        else:
            self._verified_no_collisions = True
