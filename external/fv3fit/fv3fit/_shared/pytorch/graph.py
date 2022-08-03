import torch
import tensorflow as tf
import numpy as np
import dataclasses
from hyperparameters import Hyperparameters
from toolz.functoolz import curry
from fv3fit._shared.pytorch.graph_predict import PytorchModel
from fv3fit._shared.pytorch.building_graph import graph_structure, GraphBuilder
from fv3fit._shared.pytorch.graph_config import graphnetwork, GraphNetworkConfig
from fv3fit._shared.pytorch.graph_loss import LossConfig
from fv3fit._shared.pytorch.graph_optim import OptimizerConfig
from fv3fit._shared.pytorch.training_loop import TrainingLoopConfig

# from ..tfdataset import iterable_to_tfdataset

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
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("AdamW")
    )
    build_graph: GraphBuilder = dataclasses.field(
        default_factory=lambda: GraphBuilder()
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
        return set(self.input_variables).union(self.output_variables)


# Some temporary transforms to make life easy for training process
# @curry
# def select_keys_mapping(
#     variable_names: Sequence[str], data: Mapping[str, tf.Tensor]
# ) -> Mapping[str, tf.Tensor]:
#     return {name: data[name] for name in variable_names}


@curry
def normalize(
    means: Mapping[str, float], stds: Mapping[str, float], data: Mapping[str, tf.Tensor]
) -> Mapping[str, tf.Tensor]:
    normed_data = {}
    for key, tensor in data.items():
        normed_data[key] = (tensor - means[key]) / stds[key]

    return normed_data


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
                hyperparameters.training_loop.build_samples
            )
        )
    )

    # create a a transform that will properly normalize or denormalize values
    means = apply_to_mapping(np.mean, sample)
    stds = apply_to_mapping(np.std, sample)
    processed_train_batches = processed_train_batches.map(normalize(means, stds))

    get_Xy = curry(get_Xy_dataset)(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        n_dims=3,
    )

    if validation_batches is not None:
        validation_batches = validation_batches.map(normalize(means, stds))

        val_Xy = get_Xy(data=validation_batches)
    else:
        val_Xy = None

    train_Xy = get_Xy(data=processed_train_batches)

    train_model = build_model(hyperparameters)
    optimizer = hyperparameters.optimizer_config

    hyperparameters.training_loop.fit_loop(
        hyperparameters.loss,
        train_model=train_model,
        train_data=train_Xy,
        validation=val_Xy,
        optimizer=optimizer.instance(train_model.parameters()),
        get_loss=stepwise_loss,
    )

    predictor = PytorchModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        model=train_model,
        unstacked_dims=("z",),
    )
    return predictor


def stepwise_loss(config: LossConfig, multistep, train_model, inputs, labels):

    criterion = config.loss()
    ll = 0.0
    # this is just for the identity function,
    # for prediction label would have an index over time
    for sm in range(multistep):
        if sm == 0:
            outputs = train_model(inputs)
            ll += criterion(outputs, labels)
        else:
            outputs = train_model(outputs)
            ll += criterion(outputs, labels)
    ll = ll / multistep
    return ll


def build_model(config: GraphHyperparameters):
    """
    Args:
        config: configuration of graph training
    """
    g = graph_structure(config.build_graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model = graphnetwork(config.graph_network, g).to(device)
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
