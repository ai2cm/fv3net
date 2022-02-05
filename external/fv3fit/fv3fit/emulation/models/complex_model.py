import tensorflow as tf
from tensorflow.keras.layers import Dense
from fv3fit.emulation.layers import StandardNormLayer, MeanFeatureStdDenormLayer


def build_norm_layer(sample):
    layer = StandardNormLayer()
    layer.fit(sample)
    return layer


def build_denorm_layer(sample):
    layer = MeanFeatureStdDenormLayer()
    layer.fit(sample)
    return layer


def compute_gscond(
    specific_humidity_in,
    temperature_in,
    cloud_in,
    args,
    specific_humidity_sample,
    cloud_to_vapor_sample,
    temperature_sample,
    latent_heat_sample,
    cloud_sample,
    args_sample,
):
    """

    Args:
        all shaped (-1, n) with index (i, k), k=0 is the surface

    Returns
        qv, t, qc, latent_heat
    """

    width_gscond = 256
    width_lv = 32

    qv_norm = build_norm_layer(specific_humidity_sample)(specific_humidity_in)
    qc_norm = build_norm_layer(cloud_sample)(cloud_in)
    t_norm = build_norm_layer(temperature_sample)(temperature_in)
    args_norm = [
        build_norm_layer(sample)(arg) for sample, arg in zip(args, args_sample)
    ]

    stacked = tf.stack([qv_norm, qc_norm, t_norm] + list(args_norm), axis=-1)

    cloud_to_vapor_nondim = Dense(width_gscond, activation="relu")(stacked)
    cloud_to_vapor_nondim = Dense(width_gscond, activation="relu")(
        cloud_to_vapor_nondim
    )
    cloud_to_vapor_nondim = Dense(1)(cloud_to_vapor_nondim)

    latent_heat_nondim = Dense(width_lv, activation="relu")(t_norm[:, :, None])
    latent_heat_nondim = Dense(width_lv, activation="relu")(latent_heat_nondim)
    latent_heat_nondim = Dense(1, activation="relu")(latent_heat_nondim)

    # redimensionalize
    latent_heat = build_denorm_layer(latent_heat_sample)(latent_heat_nondim)
    cloud_to_vapor = build_denorm_layer(cloud_to_vapor_sample)(cloud_to_vapor_nondim)

    # physics
    qv_out = specific_humidity_in + cloud_to_vapor
    qc_out = cloud_in - cloud_to_vapor
    t_out = temperature_in - latent_heat * cloud_to_vapor

    return qv_out, t_out, qc_out, latent_heat


# TODO de-duplicate with gscond
# TODO make compute_precpd have the same signature as gscond
def compute_precpd(
    specific_humidity_in,
    temperature_in,
    cloud_in,
    args,
    specific_humidity_sample,
    temperature_sample,
    cloud_sample,
    dqv_sample,
    dqc_sample,
    latent_heat_sample,
    args_sample,
):
    """

    Args:
        all shaped (-1, n) with index (i, k), k=0 is the surface

    Returns
        qv, t, qc, latent_heat
    """

    width_nn = 128
    width_lv = 32

    qv_norm = build_norm_layer(specific_humidity_sample)(specific_humidity_in)
    qc_norm = build_norm_layer(cloud_sample)(cloud_in)
    t_norm = build_norm_layer(temperature_sample)(temperature_in)
    args_norm = [
        build_norm_layer(sample)(arg) for sample, arg in zip(args, args_sample)
    ]

    stacked = tf.stack([qv_norm, qc_norm, t_norm] + list(args_norm), axis=-1)

    # RNN downwards dependent
    rnn1 = tf.keras.layers.SimpleRNN(width_nn, activation="relu", return_sequences=True)
    rnn2 = tf.keras.layers.SimpleRNN(2, return_sequences=True)

    reversed = stacked[:, ::-1, :]
    rnn_output = rnn2(rnn1(reversed))
    vapor_change_nondim = rnn_output[:, ::-1, 0]
    cloud_change_nondim = rnn_output[:, ::-1, 1]

    # TODO uncopy this
    latent_heat_nondim = Dense(width_lv, activation="relu")(t_norm[:, :, None])
    latent_heat_nondim = Dense(width_lv, activation="relu")(latent_heat_nondim)
    latent_heat_nondim = Dense(1, activation="relu")(latent_heat_nondim)

    # redimensionalize
    latent_heat = build_denorm_layer(latent_heat_sample)(latent_heat_nondim)
    cloud_change = build_denorm_layer(dqc_sample)(cloud_change_nondim)
    vapor_change = build_denorm_layer(dqv_sample)(vapor_change_nondim)

    # physics
    qv_out = specific_humidity_in + vapor_change
    qc_out = cloud_in + cloud_change
    t_out = temperature_in - latent_heat * vapor_change

    return qv_out, t_out, qc_out
