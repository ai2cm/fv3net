from typing import Mapping
from fv3fit.keras.adapters import ensure_dict_output
from fv3fit.emulation.transforms import TensorTransform
import tensorflow as tf

__all__ = ["transform_model"]


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
