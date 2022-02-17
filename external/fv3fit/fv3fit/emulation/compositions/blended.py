from typing import Callable, Mapping
import tensorflow as tf

from fv3fit.keras.adapters import ensure_dict_output

TensorDict = Mapping[str, tf.Tensor]


def blended_model(
    model_a: tf.keras.Model,
    model_b: tf.keras.Model,
    mask_func: Callable[[TensorDict], tf.Tensor],
    inputs: Mapping[str, tf.keras.Input],
) -> tf.keras.Model:
    """
    Take two models and blend their outputs given a mask function.

    The masking operation is broadcastable so take care to ensure
    that large tensors aren't created by omitting a feature
    dimension from the mask, when singleton feature fields exist.

    Args:
        model_a: model to use at points where mask == True
        model_b: model to use at points where mask == False
        mask_func: masking function determining model output selection.  the
            function uses the model inputs and should return a mask tensor
            with [sample, feature] or [sample, 1] dimensions
        inputs: input tensors for the model, assumed to be
            the same for model_a and model_b
    """

    model_a._name = "ModelA"
    model_b._name = "ModelB"

    # Assumes both models have exact same inputs
    # for now since training these all the same way
    # should throw error if missing info
    model_a_out = ensure_dict_output(model_a)(inputs)
    model_b_out = ensure_dict_output(model_b)(inputs)

    overlap = set(model_a_out) & set(model_b_out)

    mask = tf.cast(mask_func(inputs), tf.bool)

    new_outputs = {}
    for key in overlap:
        new_outputs[key] = tf.where(mask, x=model_a_out[key], y=model_b_out[key])

    new_outputs.update(
        {key: a_out for key, a_out in model_a_out.items() if key not in overlap}
    )
    new_outputs.update(
        {key: b_out for key, b_out in model_b_out.items() if key not in overlap}
    )

    return tf.keras.Model(inputs=inputs, outputs=new_outputs)
