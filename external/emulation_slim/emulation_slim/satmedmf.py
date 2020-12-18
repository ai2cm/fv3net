import tensorflow as tf
import numpy as np
import os
import logging

from .packer import EmuArrayPacker, consolidate_tracers, split_tracer_fields
from .debug import print_errors
from ._filesystem import get_dir


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_FILENAME = "model.tf"
X_PACKER_FILENAME = "X_packer.json"
Y_PACKER_FILENAME = "y_packer.json"

model_path = os.environ.get("TKE_EMU_MODEL")

# data for fixing del and prsi input of emulation
fix_fields = np.load("/rundir/prsi_del_fix.npz")


@print_errors
def load_model_packers(model_dir):

    with get_dir(model_dir) as path:
        
        tf_model_path = os.path.join(path, MODEL_FILENAME)
        X_packer_path = os.path.join(path, X_PACKER_FILENAME)
        y_packer_path = os.path.join(path, Y_PACKER_FILENAME)

        model = tf.keras.models.load_model(
            tf_model_path,
            custom_objects={
                "tf": tf,
                "custom_loss": None,
            }
        )

        X_packer = EmuArrayPacker.from_packer_json(X_packer_path)
        y_packer = EmuArrayPacker.from_packer_json(y_packer_path)
        logger.info(f"Loaded model and packer info from: {model_path}")

    return model, X_packer, y_packer


model, X_packer, y_packer = load_model_packers(model_path)


def get_state_stats(state):

    stats = "SATMEDMF Emulation Stats\n"
    extra_info_vars = ["dv_output", "du_output", "tdt_update"]
    for varname, arr in state.items():
        vmin = np.min(arr)
        vmax = np.max(arr)
        if varname not in extra_info_vars:
            stats += f"\t[{varname}] \tmin: {vmin:04.8f} \tmax: {vmax:04.8f}\n"
        else:
            abs_min = np.min(abs(arr[abs(arr) > 0]))            
            stats += f"\t[{varname}] \tmin: {vmin:04.8f} \tmax: {vmax:04.8f} \tabs_min: {abs_min}\n"

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


# def upper_level_delp_prsi_fix_temporary(state):
#     logger.info("Fixing upper level prsi and delp fields.")
#     prsi = state["prsi_input"]
#     delp = state["del_input"]
#     upper_slice = slice(-17, None)
#     rank = MPI.COMM_WORLD.Get_rank()
#     prsi_diff = fix_fields["prsi_input"][rank].T - prsi[upper_slice]
#     delp_diff = fix_fields["del_input"][rank].T - delp[upper_slice]
#     prsi[upper_slice] += prsi_diff
#     delp[upper_slice] += delp_diff


def add_tdt_increment(state):

    logger.info(f"Adding tdt increment from emulator...")

    tdt_orig = state["tdt_input"]
    tdt_increment = state["tdt_update"]

    logger.debug(f"tdt_orig type: {tdt_orig.dtype}")
    logger.debug(f"tdt_update type: {tdt_increment.dtype}")
    tdt_updated = tdt_orig + tdt_increment
    # Prevent IEEE_DENORMAL
    # tdt_updated[abs(tdt_updated) < 1e-28] = 0.0
    state["tdt_output"] = tdt_updated


@print_errors
def emulator(state):
    split_tracer_fields(state)
    stress_fix_temporary(state)
    logger.debug(get_state_stats(state))
    X = X_packer.to_array(state)
    logger.info("Predicting satmedmf update...")
    y = model.predict(X)
    out_state = y_packer.to_dict(y)
    consolidate_tracers(out_state)
    if "kpbl_output" in out_state:
        logger.debug("PBL index height detected in output... rounding field")
        kpbl = out_state["kpbl_output"]
        out_state["kpbl_output"] = np.round(kpbl)
    logger.info(get_state_stats(out_state))
    state.update(out_state)
    add_tdt_increment(state)
