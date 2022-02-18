from typing import Callable, Mapping
import tensorflow as tf

from fv3fit.keras.adapters import ensure_dict_output

TensorDict = Mapping[str, tf.Tensor]


def blended_model(
    model_for_true: tf.keras.Model,
    model_for_false: tf.keras.Model,
    mask_func: Callable[[TensorDict], tf.Tensor],
    inputs: Mapping[str, tf.keras.Input],
) -> tf.keras.Model:
    """
    Take two models and blend their outputs given a mask function.

    The masking operation is broadcastable so take care to ensure
    that large tensors aren't created by omitting a feature
    dimension from the mask, when singleton feature fields exist.

    Args:
        model_for_true: model to use at points where mask == True
        model_for_false: model to use at points where mask == False
        mask_func: masking function determining model output selection.  the
            function uses the blended model's inputs and should return a mask
            tensor with [sample, feature] or [sample, 1] dimensions
        inputs: input tensors for the model, assumed to be
            cover all necessary inputs for model_a and model_b
    """

    # avoid name conflicts in blended model
    model_for_true._name = "model_for_true"
    model_for_false._name = "model_for_false"

    model_for_true_out = ensure_dict_output(model_for_true)(inputs)
    model_for_false_out = ensure_dict_output(model_for_false)(inputs)

    overlap = set(model_for_true_out) & set(model_for_false_out)

    mask = tf.cast(mask_func(inputs), tf.bool)

    new_outputs = {}
    for key in overlap:
        new_outputs[key] = tf.where(
            mask, x=model_for_true_out[key], y=model_for_false_out[key]
        )

    new_outputs.update(
        {key: a_out for key, a_out in model_for_true_out.items() if key not in overlap}
    )
    new_outputs.update(
        {key: b_out for key, b_out in model_for_false_out.items() if key not in overlap}
    )

    return tf.keras.Model(inputs=inputs, outputs=new_outputs)
