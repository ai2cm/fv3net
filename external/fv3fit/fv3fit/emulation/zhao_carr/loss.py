import dataclasses
from typing import Callable, Mapping, List
from fv3fit._shared.config import OptimizerConfig
from fv3fit.emulation.losses import CustomLoss

import tensorflow as tf


@dataclasses.dataclass
class ZhaoCarrLoss:
    # dummy attr for dispatch
    zhao_carr_loss: bool = True
    optimizer: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    masked_weight: float = 2.0

    @property
    def loss_variables(self) -> List[str]:
        return [
            "air_temperature_after_gscond",
            "specific_humidity_after_gscond",
            "temperature_gscond_difference_tscaled",
            "humidity_gscond_difference_tscaled",
        ]

    def build(self, output_samples: Mapping[str, tf.Tensor]) -> Callable:
        mask_key = "nontrivial_tendency"

        loss_factory = CustomLoss(
            optimizer=self.optimizer,
            loss_variables=self.loss_variables,
            weights={
                "air_temperature_after_gscond": 100000.0,
                "specific_humidity_after_gscond": 50000.0,
            },
        )
        auto_scaled_loss = loss_factory.build(output_samples)

        def loss_masked(x, y):
            info = {}
            loss = 0.0

            mask = x[mask_key]
            loss_variables = [
                "temperature_gscond_difference_tscaled",
                "humidity_gscond_difference_tscaled",
            ]
            for v in loss_variables:
                truth = x[v]
                pred = y[v]

                mask_casted = tf.cast(mask, truth.dtype)
                loss_value = tf.keras.losses.MSE(
                    truth * mask_casted, pred * mask_casted
                )
                info[v] = loss_value
                loss += loss_value
            return loss, info

        def loss(x, y):
            lossx, infox = auto_scaled_loss(x, y)
            lossy, infoy = loss_masked(x, y)
            return lossx + lossy * self.masked_weight, {**infox, **infoy}

        return loss
