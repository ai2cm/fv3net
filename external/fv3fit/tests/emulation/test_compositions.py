import numpy as np
import pytest
import tensorflow as tf
from fv3fit.emulation.compositions import (
    apply_difference,
    blended_model,
    qc_diff,
    qv_diff,
    qv_gscond,
    t_diff,
    t_gscond,
)


def _output_layer(name):
    return tf.keras.layers.Lambda(lambda x: x, name=name)


def test_apply_precpd_difference(regtest):
    n = 5

    # inputs
    # cloud water is the only "required" input
    qc_in = tf.keras.Input(n, name="cloud_water_mixing_ratio_input")
    qv_in = tf.keras.Input(n, name="specific_humidity_input")

    model = tf.keras.Model(
        inputs=[qc_in, qv_in],
        outputs=[
            _output_layer(qv_gscond)(qc_in),
            _output_layer(t_gscond)(qc_in),
            _output_layer(t_diff)(qc_in),
            _output_layer(qc_diff)(qc_in),
            _output_layer(qv_diff)(qc_in),
        ],
    )

    model_with_after_precpd = apply_difference(model)
    outputs = model_with_after_precpd(model.inputs)
    print(sorted(outputs), file=regtest)


def _get_model(tensor_dict, factor):
    """
    Model that multiplies every tensor by a factor
    """

    in_ = {k: tf.keras.Input(v.shape[-1]) for k, v in tensor_dict.items()}
    out = tf.keras.layers.Lambda(lambda x: {k: v * factor for k, v in x.items()})(in_)
    return tf.keras.Model(inputs=in_, outputs=out)


@pytest.mark.parametrize(
    "mask_field",
    [
        tf.convert_to_tensor([[True, True], [True, False], [False, False]]),
        tf.convert_to_tensor([[True], [True], [False]]),
    ],
    ids=["full_sample_feature_mask", "sample_mask_broadcast"],
)
def test_blended_model_blending(mask_field):

    field = {"a": tf.ones((3, 2)), "mask_field": mask_field}
    model_ones = _get_model(field, 1)
    model_zeros = _get_model(field, 0)

    def mask_func(data):
        return data["mask_field"]

    inputs = {
        "a": tf.keras.Input(2),
        "mask_field": tf.keras.Input(mask_field.shape[-1]),
    }
    model = blended_model(model_ones, model_zeros, mask_func, inputs)
    result = model(field)["a"]

    expected = np.broadcast_to(mask_field, result.shape).astype(np.float)

    np.testing.assert_array_equal(result, expected)


# test that other outputs are included
def test_blended_model_extra_outputs_preserved():

    in_ = {"a": tf.keras.Input(1)}
    out1 = tf.keras.layers.Lambda(lambda x: {"b": x["a"] * 2})(in_)
    out2 = tf.keras.layers.Lambda(lambda x: {"c": x["a"] * 3})(in_)

    model1 = tf.keras.Model(in_, out1)
    model2 = tf.keras.Model(in_, out2)

    def mask_func(data):
        # not used, no overlap in outputs
        return tf.convert_to_tensor([True, False])

    model = blended_model(model1, model2, mask_func, in_)

    field_arr = tf.ones((3, 1))
    result = model({"a": field_arr})
    np.testing.assert_array_equal(result["b"], field_arr * 2)
    np.testing.assert_array_equal(result["c"], field_arr * 3)
