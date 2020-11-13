import tensorflow as tf
import numpy as np
import os
import logging

from .classifiers import _less_than_zero_mask, _less_than_equal_zero_mask
from .packer import EmuArrayPacker, consolidate_tracers, split_tracer_fields
from ._filesystem import get_dir

logger = logging.getLogger(__name__)

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


def emulator(state):
    split_tracer_fields(state)
    X = X_packer.to_array(state)
    y = model.predict(X)
    out_state = y_packer.to_dict(y)
    consolidate_tracers(out_state)
    if "kpbl_output" in out_state:
        logger.debug("PBL index height detected in output... rounding field")
        kpbl = out_state["kpbl_output"]
        out_state["kpbl_output"] = np.round(kpbl)
    state.update(out_state)
