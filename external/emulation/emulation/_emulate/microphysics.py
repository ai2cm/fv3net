import sys
from typing import Mapping
from .._typing import FortranState

# Tensorflow looks at sys args which are not initialized
# when this module is loaded under callpyfort, so ensure
# it's available here
if not hasattr(sys, "argv"):
    sys.argv = [""]

import f90nml  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import tensorflow as tf  # noqa: E402

from ..debug import print_errors  # noqa: E402
from fv3fit.keras import adapters  # noqa: E402
from .._filesystem import get_dir  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NoModel:
    """
    Dummy model to make no prediction

    Currently fv3gfs-fortran microphysics emulations
    of Zhao-Carr physics requires a model loadable to run.
    Change was introduced with piggy-backed diagnostics.
    """

    @property
    def output_names(self):
        return []

    @property
    def input_names(self):
        return []

    @staticmethod
    def predict(x):
        return {}


@print_errors
def _load_nml():
    path = os.path.join(os.getcwd(), "input.nml")
    namelist = f90nml.read(path)
    logger.info(f"Loaded namelist for ZarrMonitor from {path}")

    return namelist


@print_errors
def _get_timestep(namelist):
    return int(namelist["coupler_nml"]["dt_atmos"])


@print_errors
def _load_tf_model(model_path: str) -> tf.keras.Model:
    logger.info(f"Loading keras model: {model_path}")

    if model_path == "NO_MODEL":
        return NoModel()
    else:
        with get_dir(model_path) as local_model_path:
            model = tf.keras.models.load_model(local_model_path)
            # These following two adapters are for backwards compatibility
            dict_output_model = adapters.convert_to_dict_output(model)
            return adapters.rename_dict_output(
                dict_output_model,
                translation={
                    "air_temperature_output": "air_temperature_after_precpd",
                    "specific_humidity_output": "specific_humidity_after_precpd",
                    "cloud_water_mixing_ratio_output": "cloud_water_mixing_ratio_after_precpd",  # noqa: E501
                },
            )


class MicrophysicsHook:
    """
    Singleton class for configuring from the environment for
    the microphysics function used during fv3gfs-runtime by
    call-py-fort

    Instanced at the top level of `_emulate`
    """

    def __init__(self, model_path: str) -> None:

        self.name = "microphysics emulator"
        self.model = _load_tf_model(model_path)
        self.namelist = _load_nml()
        self.dt_sec = _get_timestep(self.namelist)
        self.orig_outputs = None

    @classmethod
    def from_environ(cls, d: Mapping):
        """
        Initialize this hook by loading configuration from environment
        variables

        Args:
            d: Mapping with key "TF_MODEL_PATH" pointing to a loadable
                keras model.  Can be local or remote.
        """

        model_path = d["TF_MODEL_PATH"]

        return cls(model_path)

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

        predictions = self.model.predict(inputs)
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
