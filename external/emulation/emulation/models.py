from typing import Callable
import logging


import numpy as np
import tensorflow as tf
from fv3fit.keras import adapters
from emulation._typing import FortranState
from emulation.zhao_carr import _get_classify_output

logger = logging.getLogger(__name__)


class ModelWithClassifier:
    def __init__(self, model, classifier=None):
        self.model = model
        self.classifier = classifier

    def __call__(self, state: FortranState):
        model_outputs = _predict(self.model, state)
        if self.classifier is not None:
            classifier_outputs = _predict(self.classifier, state)
            model_outputs.update(classifier_outputs)
        return model_outputs


class TransformedModelWithClassifier:
    def __init__(self, model, classifier):
        self.model = model
        self.classifier = classifier

    def __call__(self, state: FortranState) -> FortranState:
        transformed_inputs = _predict(self.model.forward, state)
        classes_one_hot = _predict(self.classifier, state)
        classes = _get_classify_output(classes_one_hot)
        # need to override any existing classes with the prediction
        transformed_inputs.update(classes)
        model_output = _predict(self.model.inner_model, transformed_inputs)
        transformed_inputs.update(model_output)
        return _predict(self.model.backward, transformed_inputs)


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
    try:
        regressor.get_model_inputs()
        logger.info("fv3fit.emulation.models.TransformedModel detected")
        return TransformedModelWithClassifier(regressor, classifier)
    except AttributeError:
        logger.info("tf.keras.Model detected")
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
