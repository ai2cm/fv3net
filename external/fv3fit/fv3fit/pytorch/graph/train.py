import tensorflow as tf
import numpy as np
import dataclasses
from fv3fit._shared.training_config import Hyperparameters
from toolz.functoolz import curry
from fv3fit.pytorch.predict import PytorchAutoregressor
from fv3fit.pytorch.graph.mpg_unet import MPGraphUNet, MPGUNetGraphNetworkConfig
from fv3fit.pytorch.graph.unet import GraphUNet, UNetGraphNetworkConfig

from fv3fit.pytorch.loss import LossConfig
from fv3fit.pytorch.optimizer import OptimizerConfig
from fv3fit.pytorch.training_loop import AutoregressiveTrainingConfig
from fv3fit._shared.scaler import (
    get_standard_scaler_mapping,
    get_mapping_standard_scale_func,
    StandardScaler,
)
from ..system import DEVICE

from fv3fit._shared import register_training_function
from typing import (
    Callable,
    Hashable,
    List,
    Optional,
    Sequence,
    Set,
    Mapping,
    cast,
    Union,
)
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping

network_config = Union[MPGUNetGraphNetworkConfig, UNetGraphNetworkConfig]


@dataclasses.dataclass
class GraphHyperparameters(Hyperparameters):
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
    graph_network: network_config = dataclasses.field(
        default_factory=lambda: MPGUNetGraphNetworkConfig()
    )
    training_loop: AutoregressiveTrainingConfig = dataclasses.field(
        default_factory=lambda: AutoregressiveTrainingConfig()
    )
    loss: LossConfig = LossConfig(loss_type="mse")

    @property
    def variables(self) -> Set[str]:
        return set(self.state_variables)


# TODO: Still have to handle forcing


@register_training_function("graph", GraphHyperparameters)
def train_graph_model(
    hyperparameters: GraphHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> PytorchAutoregressor:
    """
    Train a graph network.

    Args:
        hyperparameters: configuration for training
        train_batches: training data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [batch, time, tile, x, y(, z)]
        validation_batches: validation data, as a dataset of Mapping[str, tf.Tensor]
            where each tensor has dimensions [batch, time, tile, x, y(, z)]
    """
    sample = next(
        iter(train_batches.unbatch().batch(hyperparameters.normalization_fit_samples))
    )

    scalers = get_standard_scaler_mapping(sample)
    mapping_scale_func = get_mapping_standard_scale_func(scalers)

    get_state = curry(get_Xy_dataset)(
        state_variables=hyperparameters.state_variables,
        n_dims=6,  # [batch, time, tile, x, y, z]
        mapping_scale_func=mapping_scale_func,
    )

    if validation_batches is not None:
        val_state = get_state(data=validation_batches).unbatch()
    else:
        val_state = None

    train_state = get_state(data=train_batches).unbatch()

    train_model = build_model(
        hyperparameters.graph_network,
        n_state=next(iter(train_state)).shape[-1],
        nx=next(iter(train_state)).shape[3],
    )
    optimizer = hyperparameters.optimizer_config

    hyperparameters.training_loop.fit_loop(
        train_model=train_model,
        train_data=train_state,
        validation_data=val_state,
        optimizer=optimizer.instance(train_model.parameters()),
        loss_config=hyperparameters.loss,
    )

    predictor = PytorchAutoregressor(
        state_variables=hyperparameters.state_variables,
        model=train_model,
        scalers=cast(Mapping[Hashable, StandardScaler], scalers),
    )
    return predictor


def build_model(graph_network, n_state: int, nx: int):
    """
    Args:
        graph_network: configuration of the graph network
        n_state: number of state variables
    """
    if isinstance(graph_network, MPGUNetGraphNetworkConfig):
        train_model = MPGraphUNet(
            graph_network, in_channels=n_state, out_dim=n_state, nx=nx
        ).to(DEVICE)
    elif isinstance(graph_network, UNetGraphNetworkConfig):
        train_model = GraphUNet(graph_network, in_channels=n_state, out_dim=n_state).to(
            DEVICE
        )
    else:
        raise TypeError(
            "network must be either of MPGUNetGraphNetworkConfig\
            or UNetGraphNetworkConfig,"
        )
    return train_model


def get_Xy_dataset(
    state_variables: Sequence[str],
    n_dims: int,
    mapping_scale_func: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
    data: tf.data.Dataset,
):
    """
    Given a tf.data.Dataset with mappings from variable name to samples
    return a tf.data.Dataset whose entries are tensors of the requested
    state variables concatenated along the feature dimension.

    Args:
        state_variables: names of variables to include in returned tensor
        n_dims: number of dimensions of each sample, including feature dimension
        mapping_scale_func: function which scales data stored as a mapping
            from variable name to array
        data: tf.data.Dataset with mappings from variable name
            to sample tensors

    Returns:
        tf.data.Dataset where each sample is a single tensor
            containing normalized and concatenated state variables
    """
    ensure_dims = apply_to_mapping(ensure_nd(n_dims))

    def map_fn(data):
        data = mapping_scale_func(data)
        data = ensure_dims(data)
        data = select_keys(state_variables, data)
        data = tf.concat(data, axis=-1)
        return data

    return data.map(map_fn)
