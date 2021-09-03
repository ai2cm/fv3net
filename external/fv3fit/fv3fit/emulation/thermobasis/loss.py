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


def q_loss_info(truth_q, pred_q, level, timestep_seconds=900):
    """Return the specific humidity loss for ``level`` in g/kg/day"""
    secs_per_day = 86400
    g_per_kg = 1000
    factor_to_g_per_kg_per_day = g_per_kg * secs_per_day / timestep_seconds
    loss_q = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred_q))

    return {
        f"loss/variable_3/level_{level}": loss_q.numpy()
        * (factor_to_g_per_kg_per_day) ** 2
    }


class Loss(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def loss(self, prediction: Any, truth: ThermoBasis) -> Tuple[tf.Tensor, Info]:
        """Compute a loss value and corresponding information

        Returns:
            loss, info: loss is a backwards differentiable scalar value used for
                gradient descent, info is a dictionary of details about the loss
                value calculation (e.g. the MSEs of relative humidity and specific
                humidity).
        """
        pass


@dataclasses.dataclass
class QVLossSingleLevel:
    """Loss function for predicting specific humidity **at a single level**

    Attributes:
        level: the level to predict
        scale: the typical order of the loss function
    """

    level: int
    scale: float = 1.0

    def loss(self, pred_q: tf.Tensor, truth: ThermoBasis) -> Tuple[tf.Tensor, Info]:
        """

        Args:
            pred_q: the predicted specific humidity at ``self.level``.
            truth: the output state (for all levels and variables).
        """
        truth_q = select_level(truth.q, self.level)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(truth_q, pred_q))

        pred_rh = relative_humidity(
            select_level(truth.T, self.level),
            pred_q,
            select_level(truth.rho, self.level),
        )
        truth_rh = select_level(truth.rh, self.level)

        return (
            loss / self.scale,
            {
                **q_loss_info(truth_q, pred_q, self.level),
                **rh_loss_info(truth_rh, pred_rh, self.level),
            },
        )


@dataclasses.dataclass
class RHLossSingleLevel:
    """Loss function for predicting relative humidity **at a single level**

    Attributes:
        level: the level to predict
        scale: the typical order of the loss function
    """

    level: int
    scale: float = 1.0

    def loss(self, pred_rh: tf.Tensor, truth: ThermoBasis) -> Tuple[Loss, Info]:
        """
        Args:
            pred_rh: the predicted relative humidity at ``self.level``.
            truth: the output state (for all levels and variables).
        """

        pred_q = specific_humidity_from_rh(
            select_level(truth.T, self.level),
            pred_rh,
            select_level(truth.rho, self.level),
        )

        truth_q = select_level(truth.q, self.level)
        truth_rh = select_level(truth.rh, self.level)

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
        levels: levels to include in per-variable loss information returned by
            .loss. Often used for logging or evaluation.
    """

    q_weight: float = 1e6
    u_weight: float = 100
    t_weight: float = 100
    v_weight: float = 100
    rh_weight: float = 0.0
    qc_weight: float = 0.0

    levels: List[int] = dataclasses.field(default_factory=list)

    def loss(self, pred: ThermoBasis, truth: ThermoBasis) -> Tuple[Loss, Info]:
        loss_u = tf.reduce_mean(tf.keras.losses.mean_squared_error(truth.u, pred.u))
        loss_v = tf.reduce_mean(tf.keras.losses.mean_squared_error(truth.v, pred.v))
        loss_t = tf.reduce_mean(tf.keras.losses.mean_squared_error(truth.T, pred.T))
        loss_q = tf.reduce_mean(tf.keras.losses.mean_squared_error(truth.q, pred.q))
        loss_rh = tf.reduce_mean(tf.keras.losses.mean_squared_error(truth.rh, pred.rh))
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
                tf.keras.losses.mean_squared_error(truth.qc, pred.qc)
            )
            info["loss_qc"] = loss_qc.numpy()

            loss += loss_qc * self.qc_weight

        info["loss"] = loss.numpy()

        for level in self.levels:
            pred_rh = select_level(pred.rh, level)
            truth_rh = select_level(truth.rh, level)
            info.update(rh_loss_info(truth_rh, pred_rh, level))

            pred_q = select_level(pred.q, level)
            truth_q = select_level(truth.q, level)
            info.update(q_loss_info(truth_q, pred_q, level))

        return loss, info
