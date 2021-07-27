from typing import Sequence
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from runtime.emulator.thermo import (
    RelativeHumidityBasis,
    ThermoBasis,
)
from fv3fit.emulation.layers.normalization import (
    MaxFeatureStdDenormLayer,
    StandardNormLayer,
)


def atleast_2d(x: tf.Variable) -> tf.Variable:
    n = len(x.shape)
    if n == 1:
        return tf.reshape(x, shape=x.shape + [1])
    else:
        return x


def embed(args: Sequence[tf.Tensor]):
    return tf.concat([atleast_2d(arg) for arg in args], axis=-1)


def get_model(
    nz, num_scalar, num_hidden=256, num_hidden_layers=3, output_is_positive=True
):
    u = Input(shape=[None, nz])
    v = Input(shape=[None, nz])
    t = Input(shape=[None, nz])
    q = Input(shape=[None, nz])
    qc = Input(shape=[None, nz])

    inputs = [u, v, t, q, qc]

    if num_scalar > 0:
        scalars = Input(shape=[None, num_scalar])
        inputs.append(scalars)

    stacked = Concatenate()(inputs)
    for i in range(num_hidden_layers):
        stacked = Dense(num_hidden, activation="relu")(stacked)

    if output_is_positive:
        y = Dense(nz)(stacked)
    else:
        y = Dense(nz, activation="relu")(stacked)

    return tf.keras.Model(inputs=inputs, outputs=y)


class V1QCModel(tf.keras.layers.Layer):
    def __init__(self, nz, num_scalar):
        super(V1QCModel, self).__init__()

        # input scaling
        self.scale_u = StandardNormLayer()
        self.scale_v = StandardNormLayer()
        self.scale_t = StandardNormLayer()
        self.scale_rh = StandardNormLayer()
        self.scale_qc = StandardNormLayer()

        if num_scalar != 0:
            self.scale_scalars = StandardNormLayer()
        else:
            self.scale_scalars = None

        self.u_tend_model = get_model(nz, num_scalar, output_is_positive=False)
        self.u_tend_scale = MaxFeatureStdDenormLayer()
        self.v_tend_model = get_model(nz, num_scalar, output_is_positive=False)
        self.v_tend_scale = MaxFeatureStdDenormLayer()
        self.t_tend_model = get_model(nz, num_scalar, output_is_positive=False)
        self.t_tend_scale = MaxFeatureStdDenormLayer()
        self.rh_tend_model = get_model(nz, num_scalar, output_is_positive=False)
        self.rh_tend_scale = MaxFeatureStdDenormLayer()

        # qc is predicted directly
        self.qc_model = get_model(nz, num_scalar, output_is_positive=True)
        self.qc_scale = MaxFeatureStdDenormLayer()

        self.scalers_fitted = False

    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self.scale_u.fit(x.u)
        self.scale_v.fit(x.v)
        self.scale_t.fit(x.T)
        self.scale_rh.fit(x.rh)
        self.scale_qc.fit(x.qc)

        if self.scale_scalars is not None:
            self.scale_scalars.fit(embed(x.scalars))

        self.u_tend_scale.fit(y.u - x.u)
        self.v_tend_scale.fit(y.v - x.v)
        self.t_tend_scale.fit(y.T - x.T)
        self.rh_tend_scale.fit(y.rh - x.rh)
        self.qc_scale.fit(y.qc)

        self.scalers_fitted = True

    def call(self, x: ThermoBasis) -> RelativeHumidityBasis:

        inputs = [
            self.scale_u(x.u),
            self.scale_v(x.v),
            self.scale_t(x.T),
            self.scale_rh(x.rh),
            self.scale_qc(x.qc),
        ]

        if self.scale_scalars is not None:
            inputs.append(self.scale_scalars(x.scalars))

        return RelativeHumidityBasis(
            u=x.u + self.u_tend_scale(self.u_tend_model(inputs)),
            v=x.v + self.v_tend_scale(self.v_tend_model(inputs)),
            T=x.T + self.t_tend_scale(self.t_tend_model(inputs)),
            rh=x.rh + self.rh_tend_scale(self.rh_tend_model(inputs)),
            qc=self.qc_scale(self.qc_model(inputs)),
            rho=x.rho,
            dz=x.dz,
        )
