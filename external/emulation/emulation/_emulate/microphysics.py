import gc
import sys
from typing import Callable, Optional, Set

from .._typing import FortranState

# Tensorflow looks at sys args which are not initialized
# when this module is loaded under callpyfort, so ensure
# it's available here
if not hasattr(sys, "argv"):
    sys.argv = [""]

import logging  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from ..debug import print_errors  # noqa: E402
from fv3fit.keras import adapters  # noqa: E402
from .._filesystem import get_dir  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


MaskFn = Callable[[FortranState, FortranState, FortranState], FortranState]


class MicrophysicsHook:
    """
    Singleton class for configuring from the environment for
    the microphysics function used during fv3gfs-runtime by
    call-py-fort

    Instanced at the top level of `_emulate`
    """

    def __init__(
        self,
        model: tf.keras.Model,
        mask: MaskFn,
        garbage_collection_interval: int = 10,
    ) -> None:

        self.name = "microphysics emulator"

        # These following two adapters are for backwards compatibility
        dict_output_model = adapters.ensure_dict_output(model)
        self.model = adapters.rename_dict_output(
            dict_output_model,
            translation={
                "air_temperature_output": "air_temperature_after_precpd",
                "specific_humidity_output": "specific_humidity_after_precpd",
                "cloud_water_mixing_ratio_output": "cloud_water_mixing_ratio_after_precpd",  # noqa: E501
            },
        )
        self.orig_outputs: Optional[Set[str]] = None
        self.garbage_collection_interval = garbage_collection_interval
        self._calls_since_last_collection = 0
        self._mask = mask

    @staticmethod
    @print_errors
    def from_path(model_path: str, mask: MaskFn) -> tf.keras.Model:
        logger.info(f"Loading keras model: {model_path}")
        with get_dir(model_path) as local_model_path:
            model = tf.keras.models.load_model(local_model_path)
        return MicrophysicsHook(model, mask)

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
        # grab model-required variables and
        # switch state to model-expected [sample, feature]
        inputs = {name: state[name].T for name in self.model.input_names}
        inputs["latitude"] = np.rad2deg(state["latitude"].reshape((-1, 1)))

        predictions = self.model.predict(inputs)

        outputs = {name: np.atleast_2d(state[name]).T for name in predictions}

        predictions = self._mask(inputs, outputs, predictions)

        # tranpose back to FV3 conventions
        model_outputs = {name: tensor.T for name, tensor in predictions.items()}

        # fields stay in global state so check overwrites on first step
        if self.orig_outputs is None:
            self.orig_outputs = set(state).intersection(model_outputs)

        logger.info(f"Overwritting existing state fields: {self.orig_outputs}")
        microphysics_diag = {
            f"{name}_physics_diag": state[name] for name in self.orig_outputs
        }
        state.update(model_outputs)
        state.update(microphysics_diag)
        self._maybe_garbage_collect()
