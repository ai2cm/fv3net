import dataclasses
import tensorflow as tf


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

    def loss(self, model, in_, out):
        pred = model(in_)
        loss = tf.reduce_mean(
            tf.losses.mean_squared_error(
                out[self.variable][:, self.level : self.level + 1], pred
            )
        )
        return (
            loss / self.scale,
            {
                f"loss/variable_{self.variable}/level_{self.level}": loss.numpy()
                * (1000 * 86400 / 900) ** 2
            },
        )


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

    def loss(self, model, in_, out):
        up, vp, tp, qp = model(in_)
        ut, vt, tt, qt, _ = out
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
