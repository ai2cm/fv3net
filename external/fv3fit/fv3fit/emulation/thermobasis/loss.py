import abc
import dataclasses
from typing import List, Mapping, Tuple, Any
import tensorflow as tf
from fv3fit.emulation.thermobasis.thermo import ThermoBasis
from fv3fit.emulation.thermo import relative_humidity, specific_humidity_from_rh


Info = Mapping[str, float]


def rh_loss_info(truth_rh, pred_rh, level):
    loss_rh = tf.reduce_mean(tf.losses.mean_squared_error(truth_rh, pred_rh))
    return {
        f"relative_humidity_mse/level_{level}": loss_rh.numpy() * (86400 / 900) ** 2
    }


def q_loss_info(truth_q, pred_q, level):
    loss_q = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred_q))
    return {
        f"loss/variable_3/level_{level}": loss_q.numpy() * (1000 * 86400 / 900) ** 2
    }


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loss(self, prediction: Any, out: ThermoBasis) -> Tuple[tf.Tensor, Info]:
        """Compute a loss value and corresponding information

        Returns:
            loss, info: loss is a backwards differntiable scalar value used for
                gradient descent, info is a dictionary of details about the loss
                value calculation (e.g. the MSEs of relative humidity and specific
                humidity).
        """
        pass


@dataclasses.dataclass
class QVLoss(Loss):
    """Loss function for predicting specific humidity at a single level
    
    Attributes:
        level: the level to predict
        scale: the typical order of the loss function
    """

    level: int
    scale: float = 1.0

    def loss(self, pred: tf.Tensor, out: ThermoBasis) -> Tuple[tf.Tensor, Info]:
        truth_q = select_level(out.q, self.level)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred))

        pred_rh = relative_humidity(
            select_level(out.T, self.level), pred, select_level(out.rho, self.level),
        )
        truth_rh = select_level(out.rh, self.level)

        return (
            loss / self.scale,
            {
                **q_loss_info(truth_q, pred, self.level),
                **rh_loss_info(truth_rh, pred_rh, self.level),
            },
        )


@dataclasses.dataclass
class RHLoss(Loss):
    """Loss function for predicting relative humidity at a single level

    Attributes:
        variable: the variable to target, defaults to all levels of u,v,t,q
        level: the level to predict
        scale: the typical order of the loss function
    """

    level: int
    scale: float = 1.0

    def loss(self, pred_rh: tf.Tensor, out: ThermoBasis) -> Tuple[Loss, Info]:

        pred_q = specific_humidity_from_rh(
            select_level(out.T, self.level), pred_rh, select_level(out.rho, self.level),
        )

        truth_q = select_level(out.q, self.level)
        truth_rh = select_level(out.rh, self.level)

        loss_rh = tf.reduce_mean(tf.losses.mean_squared_error(truth_rh, pred_rh))

        return (
            loss_rh / self.scale,
            {
                f"loss": loss_rh.numpy(),
                **q_loss_info(truth_q, pred_q, self.level),
                **rh_loss_info(truth_rh, pred_rh, self.level),
            },
        )


def select_level(arr: tf.Tensor, level) -> tf.Tensor:
    return arr[:, level : level + 1]


@dataclasses.dataclass
class MultiVariableLoss(Loss):
    """MSE loss function with manual weights for different variables

    Attributes:
        ?_weight: weight of the variable in the loss function.
            Only used if name is None
        levels: levels to save outputs at
    """

    q_weight: float = 1e6
    u_weight: float = 100
    t_weight: float = 100
    v_weight: float = 100
    rh_weight: float = 0.0
    qc_weight: float = 0.0

    levels: List[int] = dataclasses.field(default_factory=list)

    def loss(self, pred: ThermoBasis, out: ThermoBasis) -> Tuple[Loss, Info]:
        loss_u = tf.reduce_mean(tf.keras.losses.mean_squared_error(out.u, pred.u))
        loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(out.v, pred.v))
        loss_t = tf.reduce_mean(tf.keras.losses.mean_squared_error(out.T, pred.T))
        loss_q = tf.reduce_mean(tf.keras.losses.mean_squared_error(out.q, pred.q))
        loss_rh = tf.reduce_mean(tf.keras.losses.mean_squared_error(out.rh, pred.rh))
        loss = (
            loss_u * self.u_weight
            + loss_v * self.v_weight
            + loss_t * self.t_weight
            + loss_q * self.q_weight
            + loss_rh * self.rh_weight
        )

        info = {
            "loss_u": loss_u.numpy(),
            "loss_v": loss_v.numpy(),
            "loss_q": loss_q.numpy(),
            "loss_rh": loss_rh.numpy(),
            "loss_t": loss_t.numpy(),
            "loss": loss.numpy(),
        }

        if pred.qc is not None:
            loss_qc = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(out.qc, pred.qc)
            )
            info["loss_qc"] = loss_qc.numpy()

            loss += loss_qc * self.qc_weight

        info["loss"] = loss.numpy()

        for level in self.levels:
            pred_rh = select_level(pred.rh, level)
            truth_rh = select_level(out.rh, level)
            info.update(rh_loss_info(truth_rh, pred_rh, level))

            pred_q = select_level(pred.q, level)
            truth_q = select_level(out.q, level)
            info.update(q_loss_info(truth_q, pred_q, level))

        return loss, info
