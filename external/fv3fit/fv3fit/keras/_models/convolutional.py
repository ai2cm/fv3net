import dataclasses
from typing import Hashable, List, Mapping, Optional, Sequence, Tuple, Set
from fv3fit._shared.config import (
    OptimizerConfig,
    register_training_function,
)
from fv3fit.keras._models.shared.halos import append_halos_tensor
from toolz.functoolz import curry
from fv3fit._shared.packer import clip_sample
from fv3fit.keras._models.shared.loss import LossConfig
from fv3fit.keras._models.shared.pure_keras import PureKerasModel
import tensorflow as tf
from ..._shared.config import Hyperparameters, SliceConfig
from .shared import ConvolutionalNetworkConfig, TrainingLoopConfig
import numpy as np
from fv3fit.keras._models.shared import Diffusive
import logging
from fv3fit.keras._models.shared.utils import (
    standard_denormalize,
    full_standard_normalized_input,
)
from fv3fit import tfdataset
from fv3fit.tfdataset import select_keys, ensure_nd, apply_to_mapping, apply_to_tuple

logger = logging.getLogger(__file__)

UNSTACKED_DIMS = ("x", "y", "z", "z_interface")


@dataclasses.dataclass
class ConvolutionalHyperparameters(Hyperparameters):
    """
    Args:
        input_variables: names of variables to use as inputs
        output_variables: names of variables to use as outputs
        optimizer_config: selection of algorithm to be used in gradient descent
        convolutional_network: configuration of convolutional network
        training_loop: configuration of training loop
        loss: configuration of loss functions, will be applied separately to
            each output variable
        normalization_fit_samples: number of samples to use when fitting normalization,
            note that one sample is one tile
    """

    input_variables: List[str]
    output_variables: List[str]
    optimizer_config: OptimizerConfig = dataclasses.field(
        default_factory=lambda: OptimizerConfig("Adam")
    )
    convolutional_network: ConvolutionalNetworkConfig = dataclasses.field(
        default_factory=lambda: ConvolutionalNetworkConfig()
    )
    training_loop: TrainingLoopConfig = dataclasses.field(
        default_factory=lambda: TrainingLoopConfig(batch_size=1)
    )
    clip_config: Mapping[Hashable, SliceConfig] = dataclasses.field(
        default_factory=dict
    )
    loss: LossConfig = LossConfig(scaling="standard", loss_type="mse")
    normalization_fit_samples: int = 30

    @property
    def variables(self) -> Set[str]:
        return set(self.input_variables).union(self.output_variables)


def get_Xy_dataset(
    input_variables: Sequence[str],
    output_variables: Sequence[str],
    clip_config: Optional[Mapping[Hashable, SliceConfig]],
    n_halo: int,
    data: tf.data.Dataset,
) -> tf.data.Dataset:
    """
    Given a tf.data.Dataset with mappings from variable name to samples,
    return a tf.data.Dataset whose entries are two tuples, the first containing the
    requested input variables and the second containing
    the requested output variables.
    """
    # sample, tile, x, y, z
    data = data.map(apply_to_mapping(ensure_nd(5)))

    if clip_config is not None:
        clip_function = clip_sample(clip_config)
    else:

        def clip_function(data):
            return data

    def map_fn(data):
        # clipping of inputs happens within the keras model, we don't clip at the
        # data layer so that the model still takes full-sized inputs when used
        # in production
        x = apply_to_tuple(append_halos_tensor(n_halo))(
            select_keys(input_variables, data)
        )
        y = select_keys(output_variables, clip_function(data))
        return x, y

    # unbatch to fold sample dimension into tile dimension as new sample dimension
    return data.map(map_fn).unbatch()


@register_training_function("convolutional", ConvolutionalHyperparameters)
def train_convolutional_model(
    hyperparameters: ConvolutionalHyperparameters,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset] = None,
):
    train_batches = train_batches.map(
        tfdataset.apply_to_mapping(tfdataset.float64_to_float32)
    )

    get_Xy = curry(get_Xy_dataset)(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        n_halo=hyperparameters.convolutional_network.halos_required,
    )
    if validation_batches is not None:
        validation_batches = validation_batches.map(
            tfdataset.apply_to_mapping(tfdataset.float64_to_float32)
        )
        val_Xy = get_Xy(
            clip_config=hyperparameters.clip_config, data=validation_batches
        )
    else:
        val_Xy = None

    train_Xy = get_Xy(data=train_batches, clip_config=hyperparameters.clip_config)
    # need unclipped shapes for build_model
    # have to batch data so we can take statistics for scaling transforms
    X, y = next(
        iter(
            get_Xy(data=train_batches, clip_config=None)
            .unbatch()
            .batch(hyperparameters.normalization_fit_samples)
        )
    )

    train_model, predict_model = build_model(hyperparameters, X=X, y=y)
    hyperparameters.training_loop.fit_loop(
        model=train_model, Xy=train_Xy, validation_data=val_Xy
    )
    predictor = PureKerasModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.output_variables,
        model=predict_model,
        # we train with a tile dim, but in the prognostic run each rank
        # only has local data and never has a tile dimension
        unstacked_dims=("x", "y", "z"),
        n_halo=hyperparameters.convolutional_network.halos_required,
    )
    return predictor


def build_model(
    config: ConvolutionalHyperparameters,
    X: Sequence[np.ndarray],
    y: Sequence[np.ndarray],
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Args:
        config: configuration of convolutional training
        X: example input for keras fitting, used to determine shape and normalization
        y: example output for keras fitting, used to determine shape and normalization
    """

    for item in list(X) + list(y):
        if len(item.shape) != 4:
            raise ValueError(
                "convolutional building requires 4d arrays [sample, x, y, z], "
                f"got shape {item.shape}"
            )
        if item.shape[1] != item.shape[2]:
            raise ValueError(
                "x and y dimensions should be the same length, "
                f"got shape {item.shape}"
            )
    input_layers = [tf.keras.layers.Input(shape=arr.shape[1:]) for arr in X]
    full_input = full_standard_normalized_input(input_layers, X, config.input_variables)
    convolution = config.convolutional_network.build(x_in=full_input, n_features_out=0)
    if config.convolutional_network.diffusive:
        constraint: Optional[tf.keras.constraints.Constraint] = Diffusive()
    else:
        constraint = None
    norm_output_layers = [
        tf.keras.layers.Conv2D(
            filters=array.shape[-1],
            kernel_size=(1, 1),
            padding="same",
            activation="linear",
            data_format="channels_last",
            name=f"convolutional_network_{i}_output",
            kernel_constraint=constraint,
        )(convolution.hidden_outputs[-1])
        for i, array in enumerate(y)
    ]
    denorm_output_layers = standard_denormalize(
        names=config.output_variables, layers=norm_output_layers, arrays=y,
    )
    train_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    output_stds = (
        np.std(array, axis=tuple(range(len(array.shape) - 1)), dtype=np.float32)
        for array in y
    )
    train_model.compile(
        optimizer=config.optimizer_config.instance,
        loss=[config.loss.loss(std) for std in output_stds],
    )
    # need a separate model for this so we don't have to
    # serialize the custom loss functions
    predict_model = tf.keras.Model(inputs=input_layers, outputs=denorm_output_layers)
    return train_model, predict_model
