"""Routines for backwards compatibility of model artifacts"""
from typing import Callable, Mapping, Optional, Sequence

import tensorflow as tf

TensorDict = Mapping[str, tf.Tensor]


def assoc(
    model: tf.keras.Model,
    key: str,
    func: Callable[[TensorDict, TensorDict], tf.Tensor],
    inputs: Optional[Sequence[tf.Tensor]] = None,
) -> tf.keras.Model:
    """Associate the ``key``  with the output of ``func``
    
    Args:
        model: a tensorflow model, must return a dictionary
        key: the output key
        func: the function use to compute the output
        inputs: alternative tf.keras.Input objects if ``func`` requires
            more inputs than ``model``.

    Returns:
        A new model with an output named ``key`` computed by ``func``.

    """
    inputs = inputs or model.inputs
    in_ = {key: input for key, input in zip(model.input_names, inputs)}
    out_dict = model(in_)
    out_dict[key] = func(in_, out_dict)
    model = tf.keras.Model(inputs=in_, outputs=out_dict)
    model(in_)
    return model


def _unpack_predictions(predictions, output_names):

    if len(output_names) == 1:
        # single output model doesn't return a list
        # zip would start unpacking array rows
        model_outputs = {output_names[0]: predictions}
    else:
        model_outputs = {
            name: output  # transposed adjust
            for name, output in zip(output_names, predictions)
        }

    return model_outputs


def rename_dict_output(
    model: tf.keras.Model, translation: Mapping[str, str]
) -> tf.keras.Model:
    """Rename the outputs of a dict-output model

    Args:
        model: a tensorflow model. must return dicts
        transation: a mapping from old names (the keys returned by ``model``) to
            the new output names.
    
    """

    inputs = model.inputs
    outputs = model(inputs)
    renamed_outputs = {
        translation.get(name, name): tensor for name, tensor in outputs.items()
    }
    model = tf.keras.Model(inputs=inputs, outputs=renamed_outputs)
    model(inputs)
    return model


def convert_to_dict_output(model: tf.keras.Model) -> tf.keras.Model:
    """Convert a multiple-output keras model to return dicts

    Args:
        model: a keras model returning either dicts or lists of named outputs.

    Returns:
        A keras model returning dicts.
        If ``model`` already returns dicts, then this is identical to model.
        If ``model`` outputs lists, then the ``name`` of each output
        layer will be used.
    
    Example:

        >>> i = tf.keras.Input(shape=[5])
        >>> out_1 = tf.keras.layers.Dense(5, name="a")(i)
        >>> out_2 = tf.keras.layers.Dense(5, name="b")(i)
        >>> model = tf.keras.Model(inputs=[i], outputs=[out_1, out_2])
        [<KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>, <KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>]
        >>> dict_output_model = convert_to_dict_output(model)
        >>> dict_output_model(i)
        {'a': <KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>, 'b': <KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>}

    """  # noqa: E501

    inputs = model.inputs
    outputs = model(inputs)

    if isinstance(outputs, dict):
        # model is already a dict output model
        return model

    dict_out = _unpack_predictions(outputs, model.output_names)

    model_dict_out = tf.keras.Model(inputs=inputs, outputs=dict_out)
    # need to run once to build
    model_dict_out(inputs)
    return model_dict_out
