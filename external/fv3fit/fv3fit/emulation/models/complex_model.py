from dataclasses import dataclass
from typing import Mapping
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

    width_nn = 256

    vapor_change_net = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(width_nn, activation="relu", return_sequences=True),
            tf.keras.layers.Dense(width_nn, activation="relu", return_sequences=True),
            tf.keras.layers.Dense(2, return_sequences=True),
        ]
    )

    return MoistPhysicsLayer(
        specific_humidity_sample=specific_humidity_sample,
        temperature_sample=temperature_sample,
        cloud_sample=cloud_sample,
        dqv_sample=cloud_to_vapor_sample,
        dqc_sample=-cloud_to_vapor_sample,
        latent_heat_sample=latent_heat_sample,
        args_sample=args_sample,
        vapor_change_net=vapor_change_net,
    )


def compute_precpd(
    specific_humidity_sample,
    temperature_sample,
    cloud_sample,
    dqv_sample,
    dqc_sample,
    latent_heat_sample,
    args_sample,
    width_nn: int = 128,
):
    vapor_change_net = tf.keras.Sequential(
        [
            tf.keras.layers.SimpleRNN(
                width_nn, activation="relu", return_sequences=True
            ),
            tf.keras.layers.SimpleRNN(2, return_sequences=True),
        ]
    )

    return MoistPhysicsLayer(
        specific_humidity_sample,
        temperature_sample,
        cloud_sample,
        dqv_sample,
        dqc_sample,
        latent_heat_sample,
        args_sample,
        vapor_change_net,
    )


class MoistPhysicsLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        specific_humidity_sample,
        temperature_sample,
        cloud_sample,
        dqv_sample,
        dqc_sample,
        latent_heat_sample,
        args_sample,
        vapor_change_net,
    ):
        """

        Args:
            all shaped (-1, n) with index (i, k), k=0 is the surface

        Returns
            qv, t, qc, latent_heat
        """
        super().__init__()

        width_lv = 32

        self.qv_norm_layer = build_norm_layer(specific_humidity_sample)
        self.qc_norm_layer = build_norm_layer(cloud_sample)
        self.t_norm_layer = build_norm_layer(temperature_sample)
        self.args_norm_layers = [build_norm_layer(sample) for sample in args_sample]

        # denorm layers
        self.latent_heat_denorm = build_denorm_layer(latent_heat_sample)
        self.cloud_change_denorm = build_denorm_layer(dqc_sample)
        self.vapor_change_denorm = build_denorm_layer(dqv_sample)

        self.latent_heat_net = tf.keras.Sequential(
            [
                Dense(width_lv, activation="relu"),
                Dense(width_lv, activation="relu"),
                Dense(1, activation="relu"),
            ]
        )

        self.vapor_change_net = vapor_change_net

    def call(self, specific_humidity_in, temperature_in, cloud_in, args):

        qv_norm = self.qv_norm_layer(specific_humidity_in)
        qc_norm = self.qc_norm_layer(cloud_in)
        t_norm = self.t_norm_layer(temperature_in)
        args_norm = [layer(arg) for layer, arg in zip(self.args_norm_layers, args)]

        stacked = tf.stack([qv_norm, qc_norm, t_norm] + list(args_norm), axis=-1)

        output = self.vapor_change_net(stacked)

        vapor_change_nondim = output[..., 0]
        # if second channel exists...assume it is cloud change
        if output.shape[-1] == 2:
            cloud_change_nondim = output[..., 1]
            cloud_in_output = True
        else:
            cloud_in_output = False

        latent_heat_nondim = self.latent_heat_net(t_norm[:, :, None])

        # redimensionalize
        latent_heat = self.latent_heat_denorm(latent_heat_nondim)
        vapor_change = self.vapor_change_denorm(vapor_change_nondim)
        if cloud_in_output:
            cloud_change = self.cloud_change_denorm(cloud_change_nondim)
        else:
            cloud_change = -vapor_change

        # physics
        qv_out = specific_humidity_in + vapor_change
        qc_out = cloud_in + cloud_change
        t_out = temperature_in - latent_heat * vapor_change

        return qv_out, t_out, qc_out


# some useful code
@dataclass
class Stages:
    last: str
    input: str
    gscond: str
    precpd: str

    @staticmethod
    def from_field(field: str):
        return Stages(
            field + "_after_last_gscond",
            field + "_input",
            field + "_after_gscond",
            field + "_after_precpd",
        )


@dataclass
class Inputs:
    last: tf.Tensor
    input: tf.Tensor
    gscond: tf.Tensor

    @staticmethod
    def from_stages(stages: Stages, n: int):
        return Inputs(
            tf.keras.Input(n, name=stages.last),
            tf.keras.Input(n, name=stages.input),
            tf.keras.Input(n, name=stages.gscond),
        )


qv = Stages.from_field("specific_humidity")
qc = Stages.from_field("cloud_water_mixing_ratio")
t = Stages.from_field("air_temperature")


@dataclass
class ComplexModel:
    @property
    def input_variables(self):
        pass

    def build(self, data: Mapping[str, tf.Tensor]):
        nz = data[qv.input].shape[-1]

        # make inputs
        qv_in = Inputs.from_stages(qv, nz)
        qc_in = Inputs.from_stages(qc, nz)
        t_in = Inputs.from_stages(t, nz)
        return qv_in, qc_in, t_in

        # qv_g, t_g, qc_g, lv_g = compute_gscond(
        #     qv_in.input,
        #     t_in.input,
        #     qc_in.input,
        #     args=[qv_in.last, t_in.last],
        #     specific_humidity_sample=data[qv.input],
        #     cloud_to_vapor_sample=data[qv.gscond] - data[qv.input],
        #     temperature_sample=data[t.input],
        #     latent_heat_sample=(data[t.gscond] - data[t.input])
        #     / (data[qv.gscond] - data[t.gscond]),
        #     cloud_sample=data[qc.input],
        #     args_sample=[data[qv.last], data[t.last]],
        # )

        # outputs = compute_precpd(
        #     qv_in.gscond,
        #     t_in.gscond,
        #     qc_in.gscond,
        #     args=[],
        #     specific_humidity_sample=data[qv.gscond],
        #     temperature_sample=data[t.gscond],
        #     latent_heat_sample=(data[t.precpd] - data[t.gscond])
        #     / (data[qv.precpd] - data[qv.gscond]),
        #     cloud_sample=data[qc.gscond],
        #     args_sample=[],
        # )
        # precpd_model = tf.keras.Model(
        #     inputs=[qv_in.gscond, t_in.gscond, qc_in.gscond], outputs=outputs
        # )

        # inputs_from_data = [qv_in.gscond, t_in.gscond, qc_in.gscond]
        # precpd_from_gscond = precpd_model(inputs_from_data)

        # inputs_from_gscond = [qv_g, t_g, qc_g]
        # qv_p, t_p, qc_p = precpd_model(inputs_from_gscond)
        # end_to_end_model = tf.keras.Model(
        #     inputs=[qv_in.input, t_in.input, qc_in.input, qv_in.last, t_in.last],
        #     outputs=[qv_p, t_p, qc_p, qv_g, t_g, lv_g],
        # )
