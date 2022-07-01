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

        self._forward = _predict_with_concrete_function(model.forward)
        self._backward = _predict_with_concrete_function(model.backward)
        self._inner_model = _predict_with_concrete_function(model.inner_model)

    def __call__(self, state: FortranState) -> FortranState:
        transformed_inputs = _predict(self._forward, state)
        classes_one_hot = _predict(self.classifier, state)
        classes = _get_classify_output(classes_one_hot)
        # need to override any existing classes with the prediction
        transformed_inputs.update(classes)
        model_output = _predict(self._inner_model, transformed_inputs)
        transformed_inputs.update(model_output)
        return _predict(self._backward, transformed_inputs)


def _predict_with_concrete_function(model):
    # model may be backed by several concrete functions, and depending on
    # the signatures they were created with they may produce different
    # outputs. This is a particular problem with the
    # fv3fit.emulation.transformsComposedTransform since it will silently
    # fail when tracing and one of the concrete functions will simply be the
    # identity. To work around this choose the concrete function which
    # produces the most outputs
    def number_of_outputs(f):
        (signature,), _ = f.structured_input_signature
        outs = f.structured_outputs
        return len(outs)

    def predict(state):
        inputs = {}
        for key, spec in signature.items():
            arr2d = state[key]
            tensor = tf.convert_to_tensor(arr2d)
            inputs[key] = tf.cast(tensor, spec.dtype)
        return model(inputs)

    try:
        best_f = max(model.concrete_functions, key=number_of_outputs)
        (signature,), _ = best_f.structured_input_signature
        return predict
    except AttributeError:
        return model


def _predict(model: tf.keras.Model, state: FortranState) -> FortranState:

    inputs = {name: np.atleast_2d(state[name]).T for name in state}
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
