import f90nml
import logging
import os
import tensorflow as tf

from .debug import print_errors
from ._filesystem import get_dir

logger = logging.getLogger(__name__)

# Initialized later from within functions to print errors
TF_MODEL_PATH = None
NML_PATH = None
DT_SEC = None


@print_errors
def _load_environmt_vars_into_global():

    global TF_MODEL_PATH

    cwd = os.getcwd()
    TF_MODEL_PATH = os.environ["TF_MODEL_PATH"]


@print_errors
def _load_nml():
    namelist = f90nml.read(NML_PATH)
    logger.info(f"Loaded namelist for ZarrMonitor from {NML_PATH}")
    
    global DT_SEC
    DT_SEC = int(namelist["coupler_nml"]["dt_atmos"])
    
    return namelist


@print_errors
def _load_tf_model(path: str) -> tf.keras.Model:
    local_model_path = get_dir(TF_MODEL_PATH)
    model = tf.keras.models.load_model(local_model_path)
    return model


_load_environmt_vars_into_global()
nml = _load_nml()
model = _load_tf_model(TF_MODEL_PATH)


def _get_state_variable_name(name: str) -> str:
    prefix = "tendency_of_"
    suffix = "_due_to"

    if not name.startswith(prefix) and suffix in name:
        raise ValueError(f"Received unknown tendency name format {name}")

    state_name = name[len(prefix):name.index(suffix)]
    return state_name


def _get_updated_state(state, tendency_outputs):

    state_updates = {}
    for tend_name, tendencies in tendency_outputs.items():
        state_name = _get_state_variable_name(tend_name)
        logger.debug(f"Updating state {state_name} with tendencies {tend_name}")

        input_state_var = state[state_name]
        updated_state_var = input_state_var + tendencies * DT_SEC
        state_updates[state_name] = updated_state_var

    logger.info(f"Generated state updates from tendencies {tendency_outputs.keys()}")

    return state_updates


@print_errors
def microphysics_from_tendencies(state):
    
    inputs = [state[key] for key in model.input_names]
    outputs = model.predict(inputs)
    tendency_outputs = {
        name: output
        for name, output in zip(model.output_names, outputs)
    }

    updated_state = _get_updated_state(state, tendency_outputs)
    state.update(updated_state)


@print_errors
def microphysics(state):
    
    inputs = [state[key] for key in model.input_names]
    outputs = model.predict(inputs)
    updated_state = {
        name: output
        for name, output in zip(model.output_names, outputs)
    }
    
    state.update(updated_state)
