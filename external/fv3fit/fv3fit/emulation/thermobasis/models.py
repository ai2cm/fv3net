from typing import Optional, Sequence
import tensorflow as tf
from fv3fit.emulation.thermobasis.thermo import (
    RelativeHumidityBasis,
    ThermoBasis,
    SpecificHumidityBasis,
)
from fv3fit.emulation.layers.normalization import (
    StandardNormLayer,
    StandardDenormLayer,
)

from fv3fit.emulation.thermobasis.layers import ScalarNormLayer


def atleast_2d(x: tf.Variable) -> tf.Variable:
    n = len(x.shape)
    if n == 1:
        return tf.reshape(x, shape=x.shape + [1])
    else:
        return x


class ScalarMLP(tf.keras.layers.Layer):
    def __init__(self, num_hidden=256, num_hidden_layers=1, var_level=0):
        super(ScalarMLP, self).__init__()
        self.scalers_fitted = False
        self.sequential = tf.keras.Sequential()

        # output level
        self.var_level = var_level

        # input and output normalizations
        self.norm = StandardNormLayer(name="norm")
        self.output_scaler = StandardDenormLayer(name="output_scalar")

        # model architecture
        self.sequential.add(self.norm)

        for _ in range(num_hidden_layers):
            self.sequential.add(tf.keras.layers.Dense(num_hidden, activation="relu"))

        self.sequential.add(tf.keras.layers.Dense(1, name="out"))
        self.sequential.add(self.output_scaler)

    def call(self, in_: ThermoBasis):
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        t0 = in_.q[:, self.var_level : self.var_level + 1]
        return t0 + self.sequential(stacked)

    def _fit_input_scaler(self, in_: ThermoBasis):
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def _fit_output_scaler(self, argsin: ThermoBasis, argsout: ThermoBasis):
        t0 = argsin.q[:, self.var_level : self.var_level + 1]
        t1 = argsout.q[:, self.var_level : self.var_level + 1]
        self.output_scaler.fit(t1 - t0)

    def fit_scalers(self, argsin: ThermoBasis, argsout: ThermoBasis):
        self._fit_input_scaler(argsin)
        self._fit_output_scaler(argsin, argsout)
        self.scalers_fitted = True


class RHScalarMLP(ScalarMLP):
    def fit_scalers(self, argsin: ThermoBasis, argsout: ThermoBasis):
        rh_argsin = argsin.to_rh()
        rh_argsout = argsout.to_rh()
        super(RHScalarMLP, self).fit_scalers(rh_argsin, rh_argsout)

    def call(self, args: ThermoBasis):
        rh_args = args.to_rh()
        rh = super().call(rh_args)
        return rh


class SingleVarModel(tf.keras.layers.Layer):
    def __init__(self, n, regularizer=None):
        super(SingleVarModel, self).__init__()
        self.scalers_fitted = False
        self.norm = StandardNormLayer(name="norm")
        self.linear = tf.keras.layers.Dense(
            256, name="lin", kernel_regularizer=regularizer
        )
        self.relu = tf.keras.layers.ReLU()
        self.out = tf.keras.layers.Dense(
            n, name="out", activation="relu", kernel_regularizer=regularizer
        )

        self.denorm = ScalarNormLayer()

    def _fit_input_scaler(self, args: Sequence[tf.Variable]):
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self._fit_input_scaler(x.args)
        self.denorm.fit(x.qc)
        self.scalers_fitted = True

    def call(self, in_: ThermoBasis) -> ThermoBasis:
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))
        return self.denorm(self.out(hidden))


class V1QCModel(tf.keras.layers.Layer):
    def __init__(
        self, nz, regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    ):
        super(V1QCModel, self).__init__()
        self.tend_model = UVTRHSimple(nz, nz, nz, nz, regularizer=regularizer)
        self.qc_model = SingleVarModel(nz, regularizer=regularizer)

    @property
    def scalers_fitted(self):
        return self.tend_model.scalers_fitted & self.qc_model.scalers_fitted

    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self.tend_model.fit_scalers(x, y)
        self.qc_model.fit_scalers(x, y)

    def call(self, x: ThermoBasis) -> RelativeHumidityBasis:
        y = self.tend_model(x)
        y.qc = self.qc_model(x)
        return y


class UVTQSimple(tf.keras.layers.Layer):
    def __init__(self, u_size, v_size, t_size, q_size, regularizer=None):
        super(UVTQSimple, self).__init__()
        self.scalers_fitted = False
        self.norm = StandardNormLayer(name="norm")

        self.linear = tf.keras.layers.Dense(
            256, name="lin", kernel_regularizer=regularizer
        )
        self.relu = tf.keras.layers.ReLU()
        self.out_u = tf.keras.layers.Dense(
            u_size, name="out_u", kernel_regularizer=regularizer
        )
        self.out_v = tf.keras.layers.Dense(
            v_size, name="out_v", kernel_regularizer=regularizer
        )
        self.out_t = tf.keras.layers.Dense(
            t_size, name="out_t", kernel_regularizer=regularizer
        )
        self.out_q = tf.keras.layers.Dense(
            q_size, name="out_q", kernel_regularizer=regularizer
        )

        self.scalers = [ScalarNormLayer(name=f"out_{i}") for i in range(4)]

    def _fit_input_scaler(self, args: Sequence[tf.Variable]):
        args = [atleast_2d(arg) for arg in args]
        stacked = tf.concat(args, axis=-1)
        self.norm.fit(stacked)

    def _fit_output_scaler(
        self, argsin: Sequence[tf.Variable], argsout: Sequence[tf.Variable]
    ):
        for i in range(len(self.scalers)):
            self.scalers[i].fit(argsout[i] - argsin[i])

    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self._fit_input_scaler(x.args)
        self._fit_output_scaler(x.args, y.args)
        self.scalers_fitted = True

    def call(self, in_: ThermoBasis) -> ThermoBasis:
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return SpecificHumidityBasis(
            in_.u + self.scalers[0](self.out_u(hidden)),
            in_.v + self.scalers[1](self.out_v(hidden)),
            in_.T + self.scalers[2](self.out_t(hidden)),
            in_.q + self.scalers[3](self.out_q(hidden)),
            in_.dp,
            in_.dz,
        )


class UVTRHSimple(UVTQSimple):
    def fit_scalers(self, x: ThermoBasis, y: ThermoBasis):
        self._fit_input_scaler(x.to_rh().args)
        self._fit_output_scaler(x.to_rh().args, y.to_rh().args)
        self.scalers_fitted = True

    def call(self, in_: ThermoBasis) -> RelativeHumidityBasis:
        # assume has dims: batch, z
        args = [atleast_2d(arg) for arg in in_.to_rh().args]
        stacked = tf.concat(args, axis=-1)
        hidden = self.relu(self.linear(self.norm(stacked)))

        return RelativeHumidityBasis(
            in_.u + self.scalers[0](self.out_u(hidden)),
            in_.v + self.scalers[1](self.out_v(hidden)),
            in_.T + self.scalers[2](self.out_t(hidden)),
            in_.rh + self.scalers[3](self.out_q(hidden)),
            in_.rho,
            in_.dz,
        )
