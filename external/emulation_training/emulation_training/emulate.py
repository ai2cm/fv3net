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
ONLINE_TRAIN = None
LOSSES = None
USE_PHYS_COUNTER = 0


@print_errors
def _load_environment_vars_into_global():

    global TF_MODEL_PATH
    global NML_PATH
    global ONLINE_TRAIN

    cwd = os.getcwd()
    TF_MODEL_PATH = os.environ["TF_MODEL_PATH"]
    NML_PATH = os.path.join(cwd, "input.nml")

    try:
        ONLINE_TRAIN = bool(os.environ["ONLINE_TRAIN"])
    except KeyError:
        logger.debug("Environment var ONLINE_TRAIN not set. Defaulting to False.")
        ONLINE_TRAIN = False


@print_errors
def _load_nml():
    namelist = f90nml.read(NML_PATH)
    logger.info(f"Loaded namelist for emulation from {NML_PATH}")
    
    global DT_SEC
    DT_SEC = int(namelist["coupler_nml"]["dt_atmos"])
    
    return namelist


class NormalizedMSE(tf.keras.losses.MeanSquaredError):
    """Temporary dummy"""
    
    def __init__(self, sample, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = tf.cast(tf.reduce_mean(sample, axis=0), tf.float32)
        self.sigma = tf.cast(tf.sqrt(tf.reduce_mean((sample - self.mean) ** 2)), tf.float32,)

    def _normalize(self, data):
        return (data - self.mean) / self.sigma

    def call(self, y_true, y_pred):
        return super().call(self._normalize(y_true), self._normalize(y_pred))


class DummyMSE(tf.keras.losses.MeanSquaredError):

    def call(self, y_true, y_pred):
        return 0.0


@print_errors
def _initialize_model_for_train(y):

    global LOSSES

    LOSSES = [NormalizedMSE(field) for field in y] + [DummyMSE()]
    weights = [1.0 for l in LOSSES]
    weights[-2] = 1 / 100
    MODEL.compile(loss=LOSSES, optimizer="adam", loss_weights=weights)


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

@print_errors
def _train(X, y):

    if LOSSES is None:
        _initialize_model_for_train(y[:-1])

    MODEL.fit(
        X,
        y,
        batch_size=64,
        epochs=25,
        verbose=2,
    )

@print_errors
def microphysics(state):
    global USE_PHYS_COUNTER

    inputs = [state[name].T for name in MODEL.input_names]

    if ONLINE_TRAIN:
        train_outputs = [
            state[name].T
            for name in MODEL.output_names
            if name != "total_water"
        ]
        train_outputs.append(train_outputs[-1])  # dummy for total water not included in training
        _train(inputs, train_outputs)
    
    if USE_PHYS_COUNTER == 0 or  USE_PHYS_COUNTER % 4 != 0:
        predictions = MODEL.predict(inputs)
        named_outputs = {
            name: output.T # transposed adjust
            for name, output in zip(MODEL.output_names, predictions)
        }
    else:
        named_outputs = {}

    new_state = named_outputs
    overwrites = set(state).intersection(new_state)
    logger.info(f"Overwritting existing state fields: {overwrites}")
    diag_uphys = {
        f"{orig_updated}_physics_diag": state[orig_updated]
        for orig_updated in overwrites
    }
    state.update(new_state)
    state.update(diag_uphys)

    USE_PHYS_COUNTER += 1
