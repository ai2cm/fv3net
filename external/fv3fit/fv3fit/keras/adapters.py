"""Routines for backwards compatibility of model artifacts"""
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
