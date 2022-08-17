from dataclasses import dataclass
import datetime
from typing import Callable, Optional, List
import cftime
import gc

from emulation._typing import FortranState
from emulation.masks import Mask
from emulation._time import translate_time


import logging  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def always_emulator(state: FortranState, emulator: FortranState):
    return emulator


@dataclass
class IntervalSchedule:
    """Select left value if in first half of interval of a given ``period`` and
    ``initial_time``.
    """

    period: datetime.timedelta
    initial_time: cftime.DatetimeJulian

    def __call__(self, time: cftime.DatetimeJulian) -> float:
        fraction_of_interval = ((time - self.initial_time) / self.period) % 1
        return 1.0 if fraction_of_interval < 0.5 else 0.0


@dataclass
class TimeMask:
    schedule: IntervalSchedule

    def __call__(self, state: FortranState, emulator: FortranState) -> FortranState:
        time = translate_time(state["model_time"])
        alpha = self.schedule(time)
        common_keys = set(state) & set(emulator)
        return {
            key: state[key] * alpha + emulator[key] * (1 - alpha) for key in common_keys
        }


class MicrophysicsHook:
    """Object that applies a ML model to the fortran state"""

    def __init__(
        self,
        model: Callable[[FortranState], FortranState],
        mask: Mask = always_emulator,
        garbage_collection_interval: int = 10,
    ) -> None:
        """

        Args:
            model_path: URL to model. gcs is ok too.
            mask: ``mask(state, emulator_updates)`` blends the state and
                emulator_updates into a single prediction. Used to e.g. mask
                portions of the emulators prediction.
        """

        self.name = "microphysics emulator"
        self.orig_outputs: Optional[List[str]] = None
        self.garbage_collection_interval = garbage_collection_interval
        self.mask = mask
        self._calls_since_last_collection = 0
        self.model = model

    def _maybe_garbage_collect(self):
        if self._calls_since_last_collection % self.garbage_collection_interval:
            gc.collect()
            self._calls_since_last_collection = 0
        else:
            self._calls_since_last_collection += 1

    def microphysics(self, state: FortranState) -> None:
        """
        Hook function for running the tensorflow emulator of the
        Zhao-Carr microphysics using call_py_fort.  Updates state
        dictionary in place.

        Args:
            state: Fortran state pushed into python by call_py_fort
                'set_state' calls.  Expected to be [feature, sample]
                dimensions or [sample]
        """
        model_outputs = self.model(state)

        # fields stay in global state so check overwrites on first step
        if self.orig_outputs is None:
            self.orig_outputs = sorted(set(state).intersection(model_outputs))
            logger.debug(f"Overwriting existing state fields: {self.orig_outputs}")

        microphysics_diag = {
            f"{name}_physics_diag": state[name] for name in self.orig_outputs
        }

        model_outputs.update(self.mask(state, model_outputs))
        state.update(model_outputs)
        state.update(microphysics_diag)
        self._maybe_garbage_collect()
