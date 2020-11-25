import tensorflow as tf
import numpy as np
import os
import cftime
import logging
from copy import copy

from .classifiers import _less_than_zero_mask, _less_than_equal_zero_mask
from .packer import EmuArrayPacker, consolidate_tracers, split_tracer_fields
from .monitor import output_monitor
from ._filesystem import get_dir


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_FILENAME = "model.tf"
X_PACKER_FILENAME = "X_packer.json"
Y_PACKER_FILENAME = "y_packer.json"

model_path = os.environ.get("TKE_EMU_MODEL")

with get_dir(model_path) as path:
    
    tf_model_path = os.path.join(path, MODEL_FILENAME)
    X_packer_path = os.path.join(path, X_PACKER_FILENAME)
    y_packer_path = os.path.join(path, Y_PACKER_FILENAME)

    model = tf.keras.models.load_model(
        tf_model_path,
        custom_objects={
            "tf": tf,
            "custom_loss": None,
            "_less_than_zero_mask": _less_than_zero_mask,
            "_less_than_equal_zero_mask": _less_than_equal_zero_mask,
        }
    )

    X_packer = EmuArrayPacker.from_packer_json(X_packer_path)
    y_packer = EmuArrayPacker.from_packer_json(y_packer_path)
    logger.info(f"Loaded model and packer info from: {model_path}")


def get_state_stats(state):

    stats = "SATMEDMF Emulation Stats\n"
    for varname, arr in state.items():
        vmin = np.min(arr)
        vmax = np.max(arr)
        stats += f"\t[{varname}] \tmin: {vmin:04.8f} \tmax: {vmax:04.8f}\n"

    return stats


def stress_fix_temporary(state):

    if "stress_input" not in state:
        logger.info("No stress field to check... returning")
        return

    stress = state["stress_input"]
    mask = stress < 0
    if np.any(mask):
        logger.info(f"Found negative stress at {mask.sum()} points... fixing")
        stress[stress < 0] = 0
        state["stress_input"] = stress


def add_tdt_increment(state):

    logger.info(f"Adding tdt increment from emulator...")

    tdt_orig = state["tdt_input"]
    tdt_increment = state["tdt_update"]

    logger.info(f"tdt_orig type: {tdt_orig.dtype}")
    logger.info(f"tdt_update type: {tdt_increment.dtype}")
    tdt_updated = tdt_orig + tdt_increment
    # Prevent IEEE_DENORMAL
    # tdt_updated[abs(tdt_updated) < 1e-28] = 0.0
    state["tdt_output"] = tdt_updated


def translate_time(time):
    year = time[0]
    month = time[1]
    day = time[2]
    hour = time[4]
    min = time[5]
    datetime = cftime.DatetimeJulian(year, month, day, hour, min)
    logger.info(f"Translated time: {datetime}")

    return datetime


def store_state(state, current_time):
    state["time"] = current_time
    output_monitor.store(state)
    del state["time"]


def emulator(state):
    current_time = translate_time(state.pop("model_time"))
    split_tracer_fields(state)
    stress_fix_temporary(state)
    logger.debug(get_state_stats(state))
    X = X_packer.to_array(state)
    logger.info("Predicting satmedmf update...")
    y = model.predict(X)
    out_state = y_packer.to_dict(y)
    consolidate_tracers(out_state)
    if "kpbl_output" in out_state:
        logger.info("PBL index height detected in output... rounding field")
        kpbl = out_state["kpbl_output"]
        out_state["kpbl_output"] = np.round(kpbl)
    logger.info(get_state_stats(out_state))
    state.update(out_state)
    add_tdt_increment(state)
    store_state(state, current_time)
