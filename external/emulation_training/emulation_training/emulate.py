import f90nml
import logging
import numpy as np
import os
import tensorflow as tf

from .debug import print_errors, dump_state
from ._filesystem import get_dir

logger = logging.getLogger(__name__)

# Initialized later from within functions to print errors
TF_MODEL_PATH = None
NML_PATH = None
DT_SEC = None


@print_errors
def _load_environmt_vars_into_global():

    global TF_MODEL_PATH
    global NML_PATH

    cwd = os.getcwd()
    TF_MODEL_PATH = os.environ["TF_MODEL_PATH"]
    NML_PATH = os.path.join(cwd, "input.nml")


@print_errors
def _load_nml():
    namelist = f90nml.read(NML_PATH)
    logger.info(f"Loaded namelist for emulation from {NML_PATH}")
    
    global DT_SEC
    DT_SEC = int(namelist["coupler_nml"]["dt_atmos"])
    
    return namelist


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    """Temporary dummy"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@print_errors
def _load_tf_model() -> tf.keras.Model:
    logger.debug(f"Loading keras model: {TF_MODEL_PATH}")
    with get_dir(TF_MODEL_PATH) as local_model_path:
        model = tf.keras.models.load_model(
            local_model_path, 
            custom_objects={"NormalizedMSE": NormalizedMSE}
        )

    return model


_load_environmt_vars_into_global()
NML = _load_nml()
MODEL = _load_tf_model()


def _split_tendency_state_update_predictions(predictions):

    tendency_updates = {
        name: data
        for name, data in predictions.items()
        if "tendency_of_" in name
    }

    state_update_keys = set(predictions.keys()) - set(tendency_updates.keys())
    state_updates = {
        name: predictions[name]
        for name in state_update_keys
    }

    logger.info(f"Predicted tendencies: {tendency_updates.keys()}")
    logger.info(f"Predicted states: {state_update_keys}")

    return tendency_updates, state_updates


def _get_state_variable_name(name: str) -> str:
    prefix = "tendency_of_"
    suffix = "_due_to_"

    if not name.startswith(prefix) and suffix not in name:
        raise ValueError(f"Received unknown tendency name format {name}")

    state_name = name[len(prefix):name.index(suffix)]
    return state_name


def _tendencies_to_state_update(state, tendency_outputs):

    levs = NML["fv_core_nml"]["npz"]
    state_updates = {}
    for tend_name, tendencies in tendency_outputs.items():
        state_name = _get_state_variable_name(tend_name)
        logger.debug(f"Updating state {state_name} with tendencies {tend_name}")

        in_name = state_name.replace("_output", "_input")
        input_state_var = state[in_name].T
        tendencies = _maybe_expand_field(levs, tendencies)
        updated_state_var = input_state_var + tendencies * DT_SEC
        state_updates[state_name] = updated_state_var

    return state_updates


def _combine_tend_state_updates(tend_state, direct_state):

    duplicates = set(tend_state).intersection(direct_state)
    if duplicates:
        raise ValueError(
            "State update from predictions ambiguous with overlapping"
            f" state keys {duplicates}"
        )
    
    combined = dict(direct_state)
    combined.update(tend_state)

    return combined


def _maybe_expand_field(target_lev, field):

    if field.shape[-1] < target_lev:
        expanded_data = np.zeros((field.shape[0], target_lev))
        expanded_data[:, :field.shape[-1]] = field
        field = expanded_data
    
    return field
    


# def _expand_dataset_to_full_vertical(state_predictions):
#     """
#     Hard-coded expansion for now assumes levels
#     cut off from top and all predictions are 2D
#     """
#     levs = NML["fv_core_nml"]["npz"]

#     full_levels = {}
#     for name, data in state_predictions.items():
#         full_levels[name] = _maybe_expand_field(levs, data)
    
#     logger.debug(f"Expanding feature dims for: {full_levels.keys()}")
#     new_state = dict(state_predictions)
#     new_state.update(full_levels)

#     return new_state


def _adjusted_model_inputs(state):

    adjusted = {}
    for i, name in enumerate(MODEL.input_names):
        data = state[name].T
        layer = MODEL.inputs[i]
        if layer.shape[-1] < data.shape[-1]:
            data = data[:, :layer.shape[-1]]
        adjusted[name] = data
    
    return adjusted


def _get_state_updates(state, predictions):
    
    tendencies, updated_state = \
        _split_tendency_state_update_predictions(predictions)
    
    tend_states = _tendencies_to_state_update(state, tendencies)
    combined = _combine_tend_state_updates(tend_states, updated_state)
    transposed = {name: data.T for name, data in combined.items()}
    
    logger.info(f"Predicted states from model: {transposed.keys()}")

    #add back in tendencies for debugging
    transposed.update(tendencies)

    return transposed


# TODO: quite confusing to have to keep track of transposes
#       for going from fortran -> state to numpy state
@print_errors
def microphysics(state):
    adjusted_in_state = _adjusted_model_inputs(state)
    inputs = [data for data in adjusted_in_state.values()]
    outputs = MODEL.predict(inputs)
    named_outputs = {
        name: output
        for name, output in zip(MODEL.output_names, outputs)
    }

    new_state = _get_state_updates(state, named_outputs)

    overwrites = set(state).intersection(new_state)
    logger.info(f"Overwritting existing state fields: {overwrites}")
    diag_uphys = {
        f"{orig_updated}_physics_diag": state[orig_updated]
        for orig_updated in overwrites
    }
    state.update(new_state)
    state.update(diag_uphys)
    # dump_state(state)
