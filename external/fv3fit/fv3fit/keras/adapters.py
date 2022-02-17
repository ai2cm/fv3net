"""Routines for backwards compatibility of model artifacts"""
from typing import Mapping
import tensorflow as tf


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


def ensure_dict_output(model: tf.keras.Model) -> tf.keras.Model:
    """Convert a multiple-output keras model to return dicts and have consistent
    output names

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
        >>> dict_output_model = ensure_dict_output(model)
        >>> dict_output_model(i)
        {'a': <KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>, 'b': <KerasTensor: shape=(None, 5) dtype=float32 (created by layer 'model_1')>}

    """  # noqa: E501
    return _rename_graph_outputs_to_match_output_keys(_convert_to_dict_output(model))


def get_inputs(model: tf.keras.Model) -> Mapping[str, tf.Tensor]:

    # As far as I can tell, the "input" property is not documented in tf,
    # but it does retain information about the input to the model being a dict.
    # "inputs" always seems to be a list even when a dict is used
    # to create the model

    if model.inputs and isinstance(model.input, Mapping):
        return model.input
    elif model.inputs is None:
        raise ValueError(
            f"Cannot detect inputs of model {model}. " "Custom models may not work."
        )
    else:
        return {input.name: input for input in model.inputs}


def _convert_to_dict_output(model: tf.keras.Model) -> tf.keras.Model:
    inputs = get_inputs(model)
    outputs = model(inputs)

    if isinstance(outputs, dict):
        # model is already a dict output model
        return model

    dict_out = _unpack_predictions(outputs, model.output_names)

    model_dict_out = tf.keras.Model(inputs=inputs, outputs=dict_out)
    # need to run once to build
    model_dict_out(inputs)
    return model_dict_out


def _rename_graph_outputs_to_match_output_keys(model: tf.keras.Model) -> tf.keras.Model:
    inputs = get_inputs(model)
    outputs = model(inputs)
    if set(model.output_names) == set(outputs):
        return model

    renamed = {
        key: tf.keras.layers.Lambda(lambda x: x, name=key)(val)
        for key, val in outputs.items()
    }
    model_dict_out = tf.keras.Model(inputs=inputs, outputs=renamed)
    model_dict_out(inputs)
    return model_dict_out


def _ensure_list_input(model: tf.keras.Model) -> tf.keras.Model:
    """
    Takes dict input + output model and converts it to take in
    list inputs but keep original dict outputs. Useful for connecting
    input layers with names different from the original model.
    """
    if isinstance(model.inputs, Mapping):
        inputs = [input for input in model.inputs]
    elif model.inputs is None:
        raise ValueError(
            f"Cannot detect inputs of model {model}. " "Custom models may not work."
        )
    else:
        inputs = model.inputs
    list_input_model = tf.keras.Model(inputs=inputs, outputs=model.outputs)
    list_input_model(inputs)
    # need to keep outputs dict- without calling this the model will revert
    # to list outputs
    return ensure_dict_output(list_input_model)


def rename_dict_input(
    model: tf.keras.Model, translation: Mapping[str, str]
) -> tf.keras.Model:
    """Rename the inputs of a dict-output model

    Args:
        model: a tensorflow model. must return dicts
        transation: a mapping from old names (the keys returned by ``model``) to
            the new output names.

    """

    # need to have list format inputs to connect renamed layers to rest of the model
    list_input_model = _ensure_list_input(model)
    renamed_inputs = [
        # skip the first placeholder dim when specifying shape
        tf.keras.layers.Input(
            shape=input_layer.shape[1:], name=translation.get(input_name, input_name),
        )
        for input_layer, input_name in zip(
            list_input_model.inputs, list_input_model.input_names
        )
    ]
    connect_outputs = list_input_model(renamed_inputs)
    dict_inputs = {input.name: input for input in renamed_inputs}

    model_renamed = tf.keras.Model(inputs=dict_inputs, outputs=connect_outputs)
    model_renamed(dict_inputs)

    return model_renamed


def merge_models(left: tf.keras.Model, right: tf.keras.Model) -> tf.keras.Model:
    """Merge the outputs of two models

    Args:
        left, right: the two models to merge. Both must be ``Functional`` models
            predict different outputs, and have consistent inputs.
    """
    #  combine inputs
    left = ensure_dict_output(left)
    right = ensure_dict_output(right)

    inputs = {}
    for x in left.inputs + right.inputs:
        inputs[x.name] = x

    outputs = {**left(inputs), **right(inputs)}
    outputs_renamed = {
        name: tf.keras.layers.Lambda(lambda x: x, name=name)(out)
        for name, out in outputs.items()
    }
    combined_model = tf.keras.Model(inputs=inputs, outputs=outputs_renamed)
    return combined_model
