import tensorflow as tf
import numpy as np
import dataclasses
from fv3fit._shared.training_config import Hyperparameters
from toolz.functoolz import curry
from fv3fit.pytorch.predict import PytorchModel
from fv3fit.pytorch.graph.network import GraphNetwork, GraphNetworkConfig
from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
from fv3fit.pytorch.training_loop import TrainingLoopConfig
from fv3fit._shared.scaler import StandardScaler
from ..system import DEVICE

from fv3fit._shared import register_training_function
from typing import (
    Callable,
    List,
    Optional,
    Sequence,
    Set,
    Mapping,
)
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping
from .network import define_generator, define_discriminator, define_composite_model


@dataclasses.dataclass
class CycleGANHyperparameters(Hyperparameters):
    """
    Args:
        state_variables: names of variables to evolve forward in time
        optimizer_config: selection of algorithm to be used in gradient descent
        graph_network: configuration of graph network
        training_loop: configuration of training loop
        loss: configuration of loss functions, will be applied separately to
            each output variable
    """

    state_variables: List[str]
    normalization_fit_samples: int = 50_000
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("AdamW")
    )
    graph_network: GraphNetworkConfig = dataclasses.field(
        default_factory=lambda: GraphNetworkConfig()
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self) -> Set[str]:
        return set(self.state_variables)


def train(
    d_model_A,
    d_model_B,
    g_model_AtoB,
    g_model_BtoA,
    c_model_AtoB,
    c_model_BtoA,
    dataset,
    n_batch: int,
    n_epochs: int,
):
    pass


@register_training_function("cyclegan", CycleGANHyperparameters)
def train_cyclegan(
    hyperparameters: CycleGANHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> PytorchModel:
    # define input shape based on the loaded dataset
    image_shape = dataset[0].shape[1:]
    # generator: A -> B
    g_model_AtoB = define_generator(image_shape)
    # generator: B -> A
    g_model_BtoA = define_generator(image_shape)
    # discriminator: A -> [real/fake]
    d_model_A = define_discriminator(image_shape)
    # discriminator: B -> [real/fake]
    d_model_B = define_discriminator(image_shape)
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = define_composite_model(
        g_model_AtoB, d_model_B, g_model_BtoA, image_shape
    )
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = define_composite_model(
        g_model_BtoA, d_model_A, g_model_AtoB, image_shape
    )

    # train models
    train(
        d_model_A,
        d_model_B,
        g_model_AtoB,
        g_model_BtoA,
        c_model_AtoB,
        c_model_BtoA,
        dataset,
        n_batch=1,
        n_epochs=n_epochs,
    )
