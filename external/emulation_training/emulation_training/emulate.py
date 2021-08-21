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


_load_environment_vars_into_global()
NML = _load_nml()
MODEL = _load_tf_model()


def _adjusted_model_inputs(state):

    adjusted = {}
    for i, name in enumerate(MODEL.input_names):
        data = state[name].T
        layer = MODEL.inputs[i]
        if layer.shape[-1] < data.shape[-1]:
            data = data[:, :layer.shape[-1]]
        adjusted[name] = data
    
    return adjusted


def _fix_negative(state):
    check_fields = [
        "specific_humidity_output",
        "cloud_water_mixing_ratio_output"
    ]

    for name, field in state.items():
        if name in check_fields:
            neg_locs = field < 0
            num_neg = neg_locs.sum()
            if num_neg > 0:
                logger.info(f"Fixing {num_neg} negative values to 0 in {name}")
                field[field < 0] = 0.0


# TODO: quite confusing to have to keep track of transposes
#       for going from fortran -> state to numpy state
@print_errors
def microphysics(state):
    adjusted_in_state = _adjusted_model_inputs(state)
    inputs = [data for data in adjusted_in_state.values()]
    outputs = MODEL.predict(inputs)
    named_outputs = {
        name: output.T # transposed adjust
        for name, output in zip(MODEL.output_names, outputs)
    }

    # new_state = _get_state_updates(state, named_outputs)

    new_state = named_outputs
    _fix_negative(new_state)
    overwrites = set(state).intersection(new_state)
    logger.info(f"Overwritting existing state fields: {overwrites}")
    diag_uphys = {
        f"{orig_updated}_physics_diag": state[orig_updated]
        for orig_updated in overwrites
    }
    state.update(new_state)
    state.update(diag_uphys)
    # dump_state(state)
