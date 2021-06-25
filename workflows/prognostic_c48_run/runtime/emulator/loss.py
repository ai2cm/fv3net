import dataclasses
from typing import Any, Tuple
import tensorflow as tf
from runtime.emulator.thermo import (
    ThermoBasis,
    relative_humidity,
    specific_humidity_from_rh,
)


@dataclasses.dataclass
class ScalarLoss:
    """
    Attrs:
        variable: the variable to target, defaults to all levels of u,v,t,q
        level: the level to predict
        scale: the typical order of the loss function
    """

    variable: int
    level: int
    scale: float = 1.0

    def loss(self, model, in_: ThermoBasis, out: ThermoBasis):
        # TODO remove model as an input to the loss functions
        pred = model(in_)

        truth_q = select_level(out.q, self.level)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred))

        pred_rh = relative_humidity(
            select_level(out.T, self.level), pred, select_level(out.rho, self.level),
        )
        truth_rh = select_level(out.rh, self.level)
        loss_rh = tf.reduce_mean(tf.losses.mean_squared_error(truth_rh, pred_rh))

        return (
            loss / self.scale,
            {
                f"loss/variable_{self.variable}/level_{self.level}": loss.numpy()
                * (1000 * 86400 / 900) ** 2,
                f"relative_humidity_mse/level_{self.level}": loss_rh.numpy()
                * (86400 / 900) ** 2,
            },
        )


@dataclasses.dataclass
class RHLoss:
    """
    Attrs:
        variable: the variable to target, defaults to all levels of u,v,t,q
        level: the level to predict
        scale: the typical order of the loss function
    """

    level: int
    scale: float = 1.0

    def loss(self, model, in_: ThermoBasis, out: ThermoBasis) -> Tuple[tf.Tensor, Any]:
        pred_rh = model(in_)

        pred_q = specific_humidity_from_rh(
            select_level(out.T, self.level), pred_rh, select_level(out.rho, self.level),
        )

        truth_q = select_level(out.q, self.level)
        truth_rh = select_level(out.rh, self.level)

        loss = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred_q))
        loss_rh = tf.reduce_mean(tf.losses.mean_squared_error(truth_rh, pred_rh))

        return (
            loss_rh / self.scale,
            {
                f"loss/variable_3/level_{self.level}": loss.numpy()
                * (1000 * 86400 / 900) ** 2,
                f"relative_humidity_mse/level_{self.level}": loss_rh.numpy()
                * (86400 / 900) ** 2,
                f"loss": loss_rh.numpy(),
            },
        )


def select_level(arr: tf.Tensor, level) -> tf.Tensor:
    return arr[:, level : level + 1]


@dataclasses.dataclass
class MultiVariableLoss:
    """
        Attrs:
            ?_weight: weight of the variable in the loss function.
                Only used if name is None
    """

    q_weight: float = 1e6
    u_weight: float = 100
    t_weight: float = 100
    v_weight: float = 100

    def loss(
        self, model: Any, in_: ThermoBasis, out: ThermoBasis
    ) -> Tuple[tf.Tensor, Any]:
        up, vp, tp, qp = model(in_).args[:4]
        ut, vt, tt, qt = out.args[:4]
        loss_u = tf.reduce_mean(tf.keras.losses.mean_squared_error(ut, up))
        loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(vt, vp))
        loss_t = tf.reduce_mean(tf.keras.losses.mean_squared_error(tt, tp))
        loss_q = tf.reduce_mean(tf.keras.losses.mean_squared_error(qt, qp))
        loss = (
            loss_u * self.u_weight
            + loss_v * self.v_weight
            + loss_t * self.t_weight
            + loss_q * self.q_weight
        )
        return (
            loss,
            {
                "loss_u": loss_u.numpy(),
                "loss_v": loss_v.numpy(),
                "loss_q": loss_q.numpy(),
                "loss_t": loss_t.numpy(),
                "loss": loss.numpy(),
            },
        )
