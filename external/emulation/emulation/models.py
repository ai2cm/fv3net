from typing import Callable, Any
import logging


import numpy as np
import tensorflow as tf
from fv3fit.keras import adapters
from emulation._typing import FortranState
from emulation.zhao_carr import _get_classify_output

logger = logging.getLogger(__name__)


class ModelWithClassifier:
    def __init__(self, model, classifier=None, class_key="gscond_classes"):
        self.model = model
        self.classifier = classifier
        self._class_key = class_key

    def __call__(self, state: FortranState) -> FortranState:

        if self.classifier is not None:
            classifier_outputs = _predict(self.classifier, state)
            logit_classes = classifier_outputs[self._class_key]
            classifier_outputs.update(_get_classify_output(logit_classes))
        else:
            classifier_outputs = {}

        inputs = {**classifier_outputs, **state}
        model_outputs = _predict(self.model, inputs)
        model_outputs.update(classifier_outputs)
        return model_outputs


def transform_model(
    model: Callable[[FortranState], FortranState], transform: Any
) -> Callable[[FortranState], FortranState]:
    def combined(x: FortranState) -> FortranState:
        # model time is an array of length 8, which can conflict with the other
        # arrays here
        x = {k: v for k, v in x.items() if k != "model_time"}
        x_transformed = _predict(transform.forward, x)
        x_transformed.update(model(x_transformed))
        output = _predict(transform.backward, x_transformed)
        return output

    return combined


def _predict(model: tf.keras.Model, state: FortranState) -> FortranState:
    # grab model-required variables and
    # switch state to model-expected [sample, feature]
    inputs = {name: state[name].T for name in state}

    predictions = model(inputs)
    # tranpose back to FV3 conventions
    model_outputs = {name: np.asarray(tensor).T for name, tensor in predictions.items()}

    return model_outputs


def combine_classifier_and_regressor(
    classifier, regressor
) -> Callable[[FortranState], FortranState]:
    # These following two adapters are for backwards compatibility
    dict_output_model = adapters.ensure_dict_output(regressor)
    model = adapters.rename_dict_output(
        dict_output_model,
        translation={
            "air_temperature_output": "air_temperature_after_precpd",
            "specific_humidity_output": "specific_humidity_after_precpd",
            "cloud_water_mixing_ratio_output": "cloud_water_mixing_ratio_after_precpd",  # noqa: E501
        },
    )
    return ModelWithClassifier(model, classifier)
