import torch.nn.functional as F
import tensorflow as tf
import numpy as np
from typing import Sequence
import dataclasses
from Building_Graph import BuildingGraph
from Graphloss import LossConfig
from GraphOptim import OptimizerConfig
from graph import PytorchModel
from graph_config import GraphNetworkConfig
from .hyperparameters import Hyperparameters
from toolz.functoolz import curry
from graphPredict import PytorchModel
from Building_Graph import graphStruc
from graph_config import graphnetwork
from training_loop import TrainingLoopConfig
from fv3fit.data import tfdataset
from ..._shared.config import (
    register_training_function,
)
from typing import (
    Hashable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Set,
    Mapping,
    Union,
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
        default_factory=lambda: OptimizerConfig("Adam")
    )
    build_graph: BuildingGraph = dataclasses.field(
        default_factory=lambda: BuildingGraph
    )

    graph_network: GraphNetworkConfig = dataclasses.field(
        default_factory=lambda: GraphNetworkConfig
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig(batch_size=1)
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    # normalization_fit_samples: int = 30

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


#def train(hyperparameters, train_batches, validation_batches):
    # preprocess training data
    # - normalization
    # - preparing "forcing", "input", and "output" objects (probably dicts?)

    # model = build_model(something)

    # call train_loop(model, inputs, forcing, outputs)

@register_training_function("graph", GraphHyperparameters)
def train_graph_model(
    hyperparameters: GraphHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
    input_variables: Sequence[str],
    output_variables: Sequence[str],
):

    """
    Train a graph network.

    Args:
        train_batches: training data, as a dataset of Mapping[str, tf.Tensor]
        validation_batches: validation data, as a dataset of Mapping[str, tf.Tensor]
        build_model: the function which produces a columnwise keras model
            from input and output samples. The models returned must take a list of
            tensors as input and return a list of tensors as output.
        input_variables: names of inputs for the keras model
        output_variables: names of outputs for the keras model
        clip_config: configuration of input and output clipping of last dimension
        n_loop: Total number training loop in a single epoch
        n_epoch: number of epochs
        Nbatch: number of batch size
    """

    get_Xy = curry(get_Xy_dataset)(
        input_variables=input_variables, output_variables=output_variables, n_dims=2,
    )
    
    X, y = next(
        iter(
            get_Xy(data=train_batches, clip_config=None).unbatch().batch(build_samples)
        )
    )
    """
    meanX=np.mean(X)
    calculate the normalized values 
    """
    train_model = build_model()
    optimizer = hyperparameters.optimizer_config
    

    hyperparameters.training_loop.fit_loop(train_model, train_batches, validation_batches, optimizer,get_loss=stepwise_loss)
    

    predictor = PytorchModel(
        input_variables=input_variables,
        output_variables=output_variables,
        model=train_model,
    )
    return predictor


def stepwise_loss(config: GraphHyperparameters,train_model, criterion, inputs, labels,multistep):
    # forward + backward + optimize

    criterion = config.loss.loss()
    loss=0
    for sm in range(multistep):
        if sm==0:
            outputs = train_model(inputs)
            loss += criterion(outputs, labels[sm])
        else:
            outputs = train_model(outputs)
            loss += criterion(outputs, labels[sm])
    loss=loss/multistep
    return loss


def fit_loop(train_model, Nbatch, n_epoch, n_loop,inputs, labels, optimizer, get_loss):
    for epoch in n_epoch:  # loop over the dataset multiple times
        for step in range(0,n_loop-Nbatch,Nbatch):
            optimizer.zero_grad()
            loss = get_loss(train_model, inputs, labels)                    
            loss.backward()
            optimizer.step()


def build_model(
    config: GraphHyperparameters,
):
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
        neighbor: number of nearest neighbor points around a given point
        selfpoint: True if self connected graph is needed.
    """

    # X=np.squeeze(X,0)
    # y=np.squeeze(y,0)
    # for item in list(X) + list(y):
    #     if len(item.shape) != 2:
    #         raise ValueError(
    #             "convolutional building requires 2d arrays [grids,features], "
    #             f"got shape {item.shape}"
    #         )
    #     if item.shape[1] != item.shape[2]:
    #         raise ValueError(
    #             "x and y dimensions should be the same length, "
    #             f"got shape {item.shape}"
    #         )
    g=graphStruc(config.build_graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model = graphnetwork(config.graph_network,g).to(device)
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


def get_data() -> tf.data.Dataset:
        n_records = 10
        n_batch, nt, nx, ny, nz = 10, 40, 4, 4, 6
        def records():
            for _ in range(n_records):
                record = {
                    "a": np.random.uniform([n_batch, nt, nx, ny, nz]),
                    "f": np.random.uniform([n_batch, nt, nx, ny, nz])
                }
                yield record
        tfdataset = tfdataset.from_generator(records,output_types = (tf.float32), output_shapes = (tf.TensorShape([nt, nx, ny, nz])))
        return tfdataset
