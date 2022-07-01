from dataclasses import dataclass
import datetime
import cftime
import gc
import sys

from .._typing import FortranState
from emulation.masks import Mask
from .._time import translate_time

# Tensorflow looks at sys args which are not initialized
# when this module is loaded under callpyfort, so ensure
# it's available here
if not hasattr(sys, "argv"):
    sys.argv = [""]

import logging  # noqa: E402
import tensorflow as tf  # noqa: E402

from ..debug import print_errors  # noqa: E402
from fv3fit.keras import adapters  # noqa: E402
from .._filesystem import get_dir  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _load_tf_model(model_path: str) -> tf.keras.Model:
    logger.info(f"Loading keras model: {model_path}")
    with get_dir(model_path) as local_model_path:
        return tf.keras.models.load_model(local_model_path)


@print_errors
def _load_tf_model_compatibility(model_path: str) -> tf.keras.Model:

    model = _load_tf_model(model_path)

    # These following two adapters are for backwards compatibility
    dict_output_model = adapters.ensure_dict_output(model)
    return adapters.rename_dict_output(
        dict_output_model,
        translation={
            "air_temperature_output": "air_temperature_after_precpd",
            "specific_humidity_output": "specific_humidity_after_precpd",
            "cloud_water_mixing_ratio_output": "cloud_water_mixing_ratio_after_precpd",  # noqa: E501
        },
    )


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


def always_emulator(state: FortranState, emulator: FortranState):
    return emulator


def _predict(model: tf.keras.Model, state: FortranState) -> FortranState:
    # grab model-required variables and
    # switch state to model-expected [sample, feature]
    inputs = {name: state[name].T for name in model.input_names}

    predictions = model.predict(inputs)
    # tranpose back to FV3 conventions
    model_outputs = {name: tensor.T for name, tensor in predictions.items()}

    return model_outputs


class MicrophysicsHook:
    """Object that applies a ML model to the fortran state"""

    def __init__(
        self,
        model_path: str,
        mask: Mask = always_emulator,
        garbage_collection_interval: int = 10,
        classifier_path: str = None,
    ) -> None:
        """

        Args:
            model_path: URL to model. gcs is ok too.
            mask: ``mask(state, emulator_updates)`` blends the state and
                emulator_updates into a single prediction. Used to e.g. mask
                portions of the emulators prediction.
        """

        self.name = "microphysics emulator"
        self.model = _load_tf_model_compatibility(model_path)
        self.orig_outputs = None
        self.garbage_collection_interval = garbage_collection_interval
        self.mask = mask
        self._calls_since_last_collection = 0

        self.classifier = (
            _load_tf_model(classifier_path) if classifier_path is not None else None
        )

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

        model_outputs = _predict(self.model, state)
        if self.classifier is not None:
            classifier_outputs = _predict(self.classifier, state)
            model_outputs.update(classifier_outputs)

        # fields stay in global state so check overwrites on first step
        if self.orig_outputs is None:
            self.orig_outputs = set(state).intersection(model_outputs)
            logger.debug(f"Overwriting existing state fields: {self.orig_outputs}")

        microphysics_diag = {
            f"{name}_physics_diag": state[name] for name in self.orig_outputs
        }

        model_outputs.update(self.mask(state, model_outputs))
        state.update(model_outputs)
        state.update(microphysics_diag)
        self._maybe_garbage_collect()
