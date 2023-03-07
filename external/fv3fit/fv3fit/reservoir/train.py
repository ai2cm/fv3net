import logging
import numpy as np
import tensorflow as tf
from typing import Optional, Mapping, Tuple, List


# allow reshaping of tensor data
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


from .. import Predictor
from .._shared import register_training_function, StandardScaler
from . import (
    ReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
    ReservoirComputingReadout,
    combine_readouts,
)
from .domain import RankDivider, stack_time_series_samples
from .autoencoder import Autoencoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_input_noise(arr, stddev):
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


@register_training_function("pure-reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:

    if hyperparameters.autoencoder_path is not None:
        autoencoder = Autoencoder.load(hyperparameters.autoencoder_path)
    else:
        autoencoder = None

    # Standard scale data, calculating mean and std along the stacked x,y,z,feature dim
    # This is done before passing to autoencoder
    norm_batch = next(iter(train_batches))
    norm_batch_concat = _concat_variables_along_feature_dim(norm_batch)
    scaler = StandardScaler(n_sample_dims=1)
    scaler.fit(norm_batch_concat)

    if autoencoder is not None:
        sample_batch = (
            _encode_columns(norm_batch_concat, autoencoder.encoder)
            if autoencoder is not None
            else norm_batch_concat
        )

    subdomain_config = hyperparameters.subdomain
    rank_divider = RankDivider(
        subdomain_layout=subdomain_config.layout,
        rank_dims=subdomain_config.rank_dims,
        rank_extent=tuple(sample_batch.shape),
        overlap=subdomain_config.overlap,
    )

    # first dim is time, the rest are are flattened into features
    n_input_features = int(
        np.prod(rank_divider.get_subdomain_extent(with_overlap=True)[1:])
    )

    reservoir = Reservoir(
        hyperparameters=hyperparameters.reservoir_hyperparameters,
        input_size=n_input_features,
    )

    # One readout is trained per subdomain when iterating over batches,
    # and they are combined after training into a CombinedReadout
    subdomain_readouts = [
        ReservoirComputingReadout(hyperparameters.readout_hyperparameters)
        for r in range(rank_divider.n_subdomains)
    ]

    for b, batch_data in enumerate(train_batches):
        logger.info(f"Fitting on batch {b+1}")
        # reservoir increment occurs in this call, so always call this
        # function even if X, Y are not used for readout training.
        X_batch, Y_batch = _process_batch_data(
            batch_data=batch_data,
            rank_divider=rank_divider,
            reservoir=reservoir,
            scaler=scaler,
            input_noise=hyperparameters.input_noise,
            autoencoder=autoencoder,
        )
        if b >= hyperparameters.n_batches_burn:
            _fit_readouts_over_subdomains(subdomain_readouts, X_batch, Y_batch)
    for r, readout in enumerate(subdomain_readouts):
        logger.info(
            f"Solving for readout weights: readout {r}/{len(subdomain_readouts)}"
        )
        readout.calculate_weights()

    combined_readout = combine_readouts(subdomain_readouts)
    return ReservoirComputingModel(
        input_variables=[],
        output_variables=[],
        reservoir=reservoir,
        readout=combined_readout,
        scaler=scaler,
    )


def _fit_readouts_over_subdomains(subdomain_readouts, X_batch, Y_batch):
    # Fit a readout to each subdomain's column of training data
    for i, readout in enumerate(subdomain_readouts):
        # Last dimension is subdomains
        X_subdomain = X_batch[..., i]
        Y_subdomain = Y_batch[..., i]
        readout.fit(X_subdomain, Y_subdomain, calculate_weights=False)


def _concat_variables_along_feature_dim(
    variable_tensors: Mapping[str, tf.Tensor]
) -> tf.Tensor:
    # Concat variable tensors into a single tensor along the feature dimension
    # which is assumed to be the last dim.
    return tf.concat(
        [variable_data for variable_data in variable_tensors.values()],
        axis=-1,
        name="stack",
    )


def _encode_columns(
    data, encoder,
):
    # reduce N x M x V dim data to N x M x Z dim
    # where V is original number of features (usually
    # variables * vertical levels) and Z << V is a smaller
    # number of latent dimensions
    original_sample_shape = data.shape[:-1]
    original_3d_dim = data.shape[-1]

    reshaped = data.reshape(-1, original_3d_dim)
    encoded_reshaped = encoder.predict(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)


def _process_batch_data(
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankDivider,
    reservoir: Reservoir,
    scaler: StandardScaler,
    input_noise: float,
    autoencoder: Optional[Autoencoder],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """ Obtain reservoir state corresponding to physical state,
    and reshape data into the dimensions used in training.
    """
    # Concatenate variable tensors along the feature dimension,
    # which is assumed to be the last dim.
    batch_data_concat = _concat_variables_along_feature_dim(batch_data)
    normed_batch_data_concat = scaler.normalize(batch_data_concat)

    # Convert data to latent representation if using an encoder
    if autoencoder is not None:
        logger.info("Using encoder to transform state data into latent space.")
        normed_batch_data_concat = _encode_columns(
            normed_batch_data_concat, autoencoder.encoder
        )

    # Divide into subdomains and flatten each subdomain by stacking
    # x/y/feature dims into a single feature dimension. Dimensions of a single
    # subdomain's data become [time, feature]
    X_subdomains_to_columns, Y_subdomains_to_columns = [], []
    for s in range(rank_divider.n_subdomains):
        X_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            normed_batch_data_concat, subdomain_index=s, with_overlap=True,
        )
        X_subdomains_to_columns.append(stack_time_series_samples(X_subdomain_data))

        # Prediction data does not include overlap
        Y_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            normed_batch_data_concat, subdomain_index=s, with_overlap=False,
        )
        Y_subdomains_to_columns.append(stack_time_series_samples(Y_subdomain_data))

    # Concatentate subdomain data arrays along a new subdomain axis.
    # Dimensions are now [time, feature, submdomain]
    X_reshaped = np.stack(X_subdomains_to_columns, axis=-1)
    Y_reshaped = np.stack(Y_subdomains_to_columns, axis=-1)

    # Increment and save the reservoir state after each timestep
    if reservoir.state is None:
        reservoir.reset_state(input_shape=X_reshaped[0].shape)
    reservoir_state_time_series: List[Optional[np.ndarray]] = []
    for timestep_data in X_reshaped:
        timestep_data = _add_input_noise(timestep_data, input_noise)
        print("Timestep data shape: ", timestep_data.shape)
        reservoir.increment_state(timestep_data)

        # TODO: Move the optional squaring of even hidden state terms
        # out of the readout to here
        reservoir_state_time_series.append(reservoir.state)

    # X has dimensions [time, reservoir_state, subdomain]
    X = np.array(reservoir_state_time_series[:-1])

    # Y has dimensions [time, feature, subdomain] where feature dimension
    # has flattened (x, y, z, variable) coordinates
    Y = np.array(Y_reshaped[1:])
    return X, Y
