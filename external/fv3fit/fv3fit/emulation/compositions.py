"""Model modifications requiring dataset-specific names
"""
import tensorflow as tf
from fv3fit.keras import adapters

# some harcoded names...hard to avoid since these names are baked into the dataset
t_diff = "temperature_precpd_only_difference"
qv_diff = "humidity_precpd_only_difference"
qc_diff = "cloud_precpd_difference"
t_precpd = "air_temperature_after_precpd"
t_gscond = "air_temperature_after_gscond"
qv_precpd = "specific_humidity_after_precpd"
qv_gscond = "specific_humidity_after_gscond"
qc_precpd = "cloud_water_mixing_ratio_after_precpd"
qc_in = "cloud_water_mixing_ratio_input"


def apply_precpd_difference(model):
    """Apply precpd_only and precpd differences"""

    model = adapters.ensure_dict_output(model)
    inputs = adapters.get_inputs(model)
    outputs = model(inputs)

    # apply the difference
    outputs[t_precpd] = outputs[t_gscond] + outputs[t_diff]
    outputs[qv_precpd] = outputs[qv_gscond] + outputs[qv_diff]
    outputs[qc_precpd] = inputs[qc_in] + outputs[qc_diff]

    renamed = {
        key: tf.keras.layers.Lambda(lambda x: x, name=key)(val)
        for key, val in outputs.items()
    }

    return tf.keras.Model(inputs=inputs, outputs=renamed)
