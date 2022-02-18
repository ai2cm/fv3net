import tensorflow as tf
from fv3fit.emulation.compositions import (
    apply_precpd_difference,
    t_diff,
    qc_diff,
    qv_diff,
    t_gscond,
    qv_gscond,
)


def _output_layer(name):
    return tf.keras.layers.Lambda(lambda x: x, name=name)


def test_apply_precpd_difference(regtest):
    n = 5

    # inputs
    # cloud water is the only "required" input
    qc_in = tf.keras.Input(n, name="cloud_water_mixing_ratio_input")

    model = tf.keras.Model(
        inputs=[qc_in],
        outputs=[
            _output_layer(qv_gscond)(qc_in),
            _output_layer(t_gscond)(qc_in),
            _output_layer(t_diff)(qc_in),
            _output_layer(qc_diff)(qc_in),
            _output_layer(qv_diff)(qc_in),
        ],
    )

    model_with_after_precpd = apply_precpd_difference(model)
    outputs = model_with_after_precpd(model.inputs)
    print(sorted(outputs), file=regtest)
