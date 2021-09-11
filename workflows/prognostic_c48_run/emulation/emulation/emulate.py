import sys
if not hasattr(sys, "argv"):
    sys.argv = [""]

import f90nml
import logging
import os
import tensorflow as tf

from .debug import print_errors
from ._filesystem import get_dir

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TF_MODEL_PATH = None  # local or remote path to tensorflow model
NML_PATH = None
DT_SEC = None
ORIG_OUTPUTS = None

@print_errors
def _load_environment_vars_into_global():

    global TF_MODEL_PATH
    global NML_PATH

    cwd = os.getcwd()
    TF_MODEL_PATH = os.environ["TF_MODEL_PATH"]
    NML_PATH = os.path.join(cwd, "input.nml")


@print_errors
def _load_nml():
    namelist = f90nml.read(NML_PATH)
    logger.info(f"Loaded namelist for emulation from {NML_PATH}")
    
    return namelist

@print_errors
def _get_timestep(namelist):
    return int(namelist["coupler_nml"]["dt_atmos"])


@print_errors
def _load_tf_model() -> tf.keras.Model:
    logger.debug(f"Loading keras model: {TF_MODEL_PATH}")
    with get_dir(TF_MODEL_PATH) as local_model_path:
        model = tf.keras.models.load_model(local_model_path)

    return model


_load_environment_vars_into_global()
NML = _load_nml()
DT_SEC = _get_timestep(NML)
MODEL = _load_tf_model()


def _unpack_predictions(predictions):

    if len(MODEL.output_names) == 1:
        # single output model doesn't return a list
        # zip would start unpacking array rows
        model_outputs = {MODEL.output_names[0]: predictions.T}
    else:
        model_outputs = {
            name: output.T # transposed adjust
            for name, output in zip(MODEL.output_names, predictions)
        }

    return model_outputs


@print_errors
def microphysics(state):

    global ORIG_OUTPUTS

    inputs = [state[name].T for name in MODEL.input_names]
    predictions = MODEL.predict(inputs)
    model_outputs = _unpack_predictions(predictions)

    # fields stay in global state so check overwrites on first step
    if ORIG_OUTPUTS is None:
        ORIG_OUTPUTS = set(state).intersection(model_outputs)

    logger.info(f"Overwritting existing state fields: {ORIG_OUTPUTS}")
    microphysics_diag = {
        f"{name}_physics_diag": state[name]
        for name in ORIG_OUTPUTS
    }
    state.update(model_outputs)
    state.update(microphysics_diag)
