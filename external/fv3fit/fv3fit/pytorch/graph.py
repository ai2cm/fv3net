import torch
import tensorflow as tf
import numpy as np
import dgl
import dataclasses
from .._shared.training_config import Hyperparameters
from toolz.functoolz import curry
from fv3fit.pytorch.graph_predict import PytorchModel
from fv3fit.pytorch.graph_builder import build_graph, GraphConfig
from fv3fit.pytorch.graph_config import GraphNetwork, GraphNetworkConfig
from fv3fit.pytorch.graph_loss import LossConfig
from fv3fit.pytorch.graph_optim import OptimizerConfig
from fv3fit.pytorch.training_loop import TrainingLoopConfig
from fv3fit._shared.scaler import StandardScaler

from fv3fit._shared import register_training_function
from typing import (
    List,
    Optional,
    Sequence,
    Set,
    Mapping,
)
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping


@dataclasses.dataclass
class GraphHyperparameters(Hyperparameters):
    """
    Args:
        input_variables: names of variables to use as inputs
        output_variables: names of variables to use as outputs
        optimizer_config: selection of algorithm to be used in gradient descent
        graph_network: configuration of graph network
        training_loop: configuration of training loop
        loss: configuration of loss functions, will be applied separately to
            each output variable
    """

    input_variables: List[str]
    output_variables: List[str]
    normalization_fit_samples: int = 50_000
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("AdamW")
    )
    build_graph: GraphConfig = dataclasses.field(default_factory=lambda: GraphConfig())

    graph_network: GraphNetworkConfig = dataclasses.field(
        default_factory=lambda: GraphNetworkConfig()
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


def get_normalizer(sample: Mapping[str, np.ndarray]):
    scalers = {}
    for name, array in sample.items():
        s = StandardScaler()
        s.fit(array)
        scalers[name] = s

    def scale(sample: Mapping[str, np.ndarray]):
        output = {**sample}
        for name, array in sample.items():
            output[name] = scalers[name].normalize(array)
        return output

    return scale


# TODO: Still have to handle forcing


@register_training_function("graph", GraphHyperparameters)
def train_graph_model(
    hyperparameters: GraphHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
):

    """
    Train a graph network.

    Args:
        train_batches: training data, as a dataset of Mapping[str, tf.Tensor]
        validation_batches: validation data, as a dataset of Mapping[str, tf.Tensor]
        build_model: the function which produces the pytorch model
            from input and output samples. The models returned must take a list of
            tensors as input and return a list of tensors as output.
        input_variables: names of inputs for the pytorch model
        output_variables: names of outputs for the pytorch model
        n_epoch: number of epochs
    """

    # use transforms to get correct variables, correct dimensions
    processed_train_batches = train_batches.map(apply_to_mapping(ensure_nd(3)))

    sample = next(
        iter(
            processed_train_batches.unbatch().batch(
                hyperparameters.normalization_fit_samples
            )
        )
    )

    normalizer = get_normalizer(sample)

    get_Xy = curry(get_Xy_dataset)(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        n_dims=3,
    )

    if validation_batches is not None:
        validation_batches = validation_batches.map(apply_to_mapping(ensure_nd(3)))
        validation_batches = validation_batches.map(normalizer)
        val_Xy = get_Xy(data=validation_batches)
    else:
        val_Xy = None

    processed_train_batches = processed_train_batches.map(normalizer)
    train_Xy = get_Xy(data=processed_train_batches)

    train_model = build_model(hyperparameters)
    optimizer = hyperparameters.optimizer_config

    hyperparameters.training_loop.fit_loop(
        hyperparameters.loss,
        train_model=train_model,
        train_data=train_Xy,
        validation_data=val_Xy,
        optimizer=optimizer.instance(train_model.parameters()),
    )

    predictor = PytorchModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        model=train_model,
        unstacked_dims=("z",),
    )
    return predictor


def build_model(config: GraphHyperparameters):
    """
    Args:
        config: configuration of graph training
    """
    graph_data = build_graph(config.build_graph)
    g = dgl.graph(graph_data)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model = GraphNetwork(
        config.graph_network, g.to(device), n_features_in=2, n_features_out=2
    ).to(device)
    return train_model


def get_Xy_dataset(
    input_variables: Sequence[str],
    output_variables: Sequence[str],
    n_dims: int,
    data: tf.data.Dataset,
):
    """
    Given a tf.data.Dataset with mappings from variable name to samples,
    return a tf.data.Dataset whose entries are two tuples, the first containing the
    requested input variables and the second containing
    the requested output variables.
    """
    data = data.map(apply_to_mapping(ensure_nd(n_dims)))

    def map_fn(data):
        x = select_keys(input_variables, data)
        y = select_keys(output_variables, data)
        return x, y

    return data.map(map_fn)
