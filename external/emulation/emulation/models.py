from typing import Callable, Any, Optional, Sequence
import logging


import numpy as np
import tensorflow as tf
from fv3fit.keras import adapters
from emulation._typing import FortranState
from emulation.zhao_carr import _get_classify_output

logger = logging.getLogger(__name__)


class ModelWithClassifier:
    def __init__(
        self,
        model: tf.keras.Model,
        classifier: Optional[tf.keras.Model] = None,
        class_key: str = "gscond_classes",
        batch_size: int = 1024,
        inputs_to_ignore: Sequence[str] = ("rank", "model_time"),
    ):
        """
        Args:
            inputs_to_ignore: variables that ``model`` and ``classifier`` may
                reject. For example, scalar or singleton value variables.
        """
        self.model = adapters.ensure_dict_output(model)
        if classifier is None:
            self.classifier = None
        else:
            self.classifier = adapters.ensure_dict_output(classifier)
        self._class_key = class_key
        self._batch_size = batch_size
        self.inputs_to_ignore = inputs_to_ignore

    def __call__(self, state: FortranState) -> FortranState:
        state = {k: v for k, v in state.items() if k not in self.inputs_to_ignore}

        if self.classifier is not None:
            classifier_outputs = self.classifier.predict(
                state, batch_size=self._batch_size
            )
            logit_classes = classifier_outputs[self._class_key]
            decoded_one_hot = _get_classify_output(logit_classes, one_hot_axis=-1)
            classifier_outputs.update(decoded_one_hot)
        else:
            classifier_outputs = {}

        inputs = {**classifier_outputs, **state}
        model_outputs = self.model.predict(inputs, batch_size=self._batch_size)
        model_outputs.update(classifier_outputs)
        return _as_numpy(model_outputs)


def transform_model(
    model: Callable[[FortranState], FortranState], transform: Any
) -> Callable[[FortranState], FortranState]:
    def combined(x: FortranState) -> FortranState:
        x_transformed = transform.forward(x)
        x_transformed.update(model(x_transformed))
        output = transform.backward(x_transformed)
        return output

    return combined


def _as_numpy(state):
    return {key: np.asarray(state[key]) for key in state}


def combine_classifier_and_regressor(
    classifier, regressor, batch_size: int
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
    return ModelWithClassifier(model, classifier, batch_size=batch_size)
