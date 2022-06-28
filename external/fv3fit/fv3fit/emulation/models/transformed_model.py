from typing import Mapping
from fv3fit.keras.adapters import ensure_dict_output
from fv3fit.emulation.transforms import TensorTransform
from fv3fit.emulation.types import TensorDict
import tensorflow as tf

__all__ = ["transform_model", "TransformedModel"]


class TransformedModel(tf.keras.Model):
    """A serializeable module for with methods for transforms and inner model

    Useful in circumstances when we want to override the output of a
    transform....e.g. with another machine learning model.

    Example:

        >>> y = transformed_model.forward(x)
        >>> z = some_other_model(x)
        >>> merged={**y, **z}
        >>> out = transformed_model.inner_model(merged)
        >>> final_out = transformed_model.backward(out)

    """

    def __init__(
        self, model: tf.keras.Model, transform: TensorTransform, inputs: TensorDict
    ):
        """
        Args:
            model: a keras functional model, must have model.input_names attribute
        """
        super().__init__()
        self.model = model
        self.transform = transform
        self.inputs = inputs

    @tf.function
    def forward(self, x):
        """Transform the data forward

        Note:
            This function is serialized
        """
        return self.transform.forward(x)

    @tf.function
    def backward(self, x):
        """Transform the data backward

        Note:
            This function is serialized
        """
        return self.transform.backward(x)

    @tf.function
    def inner_model(self, x):
        """Call the inner model without the forwards or backwards transformations

        Note:
            This function is serialized
        """
        return self.model(x)

    @tf.function
    def call(self, x):
        """Call the combined transform model

        Note:
            This function is serialized
        """
        return self.backward(self.inner_model(self.forward(x)))

    @tf.function
    def get_model_inputs(self):
        """Return a list of the inner model input names

        Note:
            This function is serialized
        """
        return list(self.model.input_names)

    def as_functional_model(self):
        return transform_model(self.model, self.transform, self.inputs)

    def save(self, path: str, save_format: str = "tf"):
        assert save_format == "tf"

        def _get_concrete(f, x: TensorDict):
            return f.get_concrete_function(
                {name: tf.TensorSpec(i.shape, dtype=i.dtype) for name, i in x.items()}
            )

        backward_ins = self.transform.forward(self.inputs)
        model_ins = {
            name: tensor
            for name, tensor in backward_ins.items()
            if name in self.model.input_names
        }
        # call once to avoid error
        self(self.inputs)
        super().save(
            path,
            signatures={
                "forward": _get_concrete(self.forward, self.inputs),
                "inner_model": _get_concrete(self.inner_model, model_ins),
                "backward": _get_concrete(self.backward, backward_ins),
                "get_model_inputs": self.get_model_inputs.get_concrete_function(),
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: _get_concrete(
                    self.backward, model_ins
                ),
            },
        )


def transform_model(
    model: tf.keras.Model,
    transform: TensorTransform,
    inputs: Mapping[str, tf.keras.Input],
) -> tf.keras.Model:
    try:
        model = ensure_dict_output(model)
    except ValueError:
        pass
    # Wrap the custom model with a keras functional model for easier
    # serialization. Serialized models need to know their input/output
    # signatures. The keras "Functional" API makes this explicit, but custom
    # models subclasses "remember" their first inputs. Since ``data``
    # contains both inputs and outputs the serialized model will think its
    # outputs are also inputs and never be able to evaluate...even though
    # calling `model(data)` works just fine.
    outputs = model(transform.forward(inputs))

    # combine inputs and outputs for the reverse transformation some
    # transformations (e.g. residual) depend on outputs and inputs
    out_and_in = {**inputs}
    out_and_in.update(outputs)
    outputs = transform.backward(out_and_in)

    # filter out inputs that were unchanged by the transform
    new_outputs = {
        key: tensor for key, tensor in outputs.items() if (key not in inputs)
    }

    functional_keras_model = tf.keras.Model(inputs=inputs, outputs=new_outputs)
    return ensure_dict_output(functional_keras_model)
