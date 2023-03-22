import logging
from fv3fit.reservoir.readout import BatchLinearRegressor
import numpy as np
import tensorflow as tf
from typing import Optional, Mapping, Tuple, List, Iterable

from .. import Predictor
from .utils import square_even_terms
from .autoencoder import Autoencoder
from .._shared import register_training_function
from . import (
    ReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
    ReservoirComputingReadout,
)
from .readout import combine_readouts
from .domain import (
    RankDivider,
    stack_time_series_samples,
    concat_variables_along_feature_dim,
)

# allow reshaping of tensor data
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_input_noise(arr: np.ndarray, stddev: float) -> np.ndarray:
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


def _encode_columns(data: tf.Tensor, encoder: tf.keras.Model,) -> np.ndarray:
    # reduce N x M x V dim data to N x M x Z dim
    # where V is original number of features (usually
    # variables * vertical levels) and Z << V is a smaller
    # number of latent dimensions
    original_sample_shape = data.shape[:-1]
    original_z_dim = data.shape[-1]

    reshaped = data.reshape(-1, original_z_dim)
    encoded_reshaped = encoder.predict(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)


@register_training_function("pure-reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:

    if hyperparameters.autoencoder_path is not None:
        autoencoder = Autoencoder.load(hyperparameters.autoencoder_path)
    else:
        autoencoder = None  # type: ignore

    sample_batch = next(iter(train_batches))
    sample_batch_concat = concat_variables_along_feature_dim(
        variables=hyperparameters.input_variables, variable_tensors=sample_batch
    )

    subdomain_config = hyperparameters.subdomain
    rank_extent = (
        [*sample_batch_concat.shape[:1], autoencoder.n_latent_dims]
        if autoencoder is not None
        else sample_batch_concat.shape
    )
    rank_divider = RankDivider(
        subdomain_layout=subdomain_config.layout,
        rank_dims=subdomain_config.rank_dims,
        rank_extent=rank_extent,
        overlap=subdomain_config.overlap,
    )

    # First data dim is time, the rest of the elements of each
    # subdomain+halo are are flattened into feature dimension
    reservoir = Reservoir(
        hyperparameters=hyperparameters.reservoir_hyperparameters,
        input_size=rank_divider.n_subdomain_features,
    )

    # One readout is trained per subdomain when iterating over batches,
    # and they are combined after training into a CombinedReadout
    subdomain_regressors = [
        BatchLinearRegressor(hyperparameters.readout_hyperparameters)
        for r in range(rank_divider.n_subdomains)
    ]
    for b, batch_data in enumerate(train_batches):
        time_series_with_overlap, time_series_without_overlap = _process_batch_data(
            variables=hyperparameters.input_variables,
            batch_data=batch_data,
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
        if b < hyperparameters.n_batches_burn:
            logger.info(f"Synchronizing on batch {b+1}")

        # reservoir increment occurs in this call, so always call this
        # function even if X, Y are not used for readout training.
        reservoir_state_time_series = _get_reservoir_state_time_series(
            time_series_with_overlap, hyperparameters.input_noise, reservoir
        )

        if b >= hyperparameters.n_batches_burn:
            logger.info(f"Fitting on batch {b+1}")
            _fit_batch_over_subdomains(
                subdomain_regressors=subdomain_regressors,
                reservoir_state_time_series=reservoir_state_time_series,
                prediction_time_series=time_series_without_overlap,
                square_even_inputs=hyperparameters.square_half_hidden_state,
            )

    subdomain_readouts = []
    for r, regressor in enumerate(subdomain_regressors):
        logger.info(
            f"Solving for readout weights: readout {r+1}/{len(subdomain_regressors)}"
        )
        coefs_, intercepts_ = regressor.get_weights()
        subdomain_readouts.append(
            ReservoirComputingReadout(coefficients=coefs_, intercepts=intercepts_)
        )
    readout = combine_readouts(subdomain_readouts)

    model = ReservoirComputingModel(
        input_variables=hyperparameters.input_variables,
        output_variables=hyperparameters.input_variables,
        reservoir=reservoir,
        readout=readout,
        square_half_hidden_state=hyperparameters.square_half_hidden_state,
        rank_divider=rank_divider,
    )
    return model


def _process_batch_data(
    variables: Iterable[str],
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankDivider,
    autoencoder: Optional[Autoencoder],
) -> Tuple[np.ndarray, np.ndarray]:
    """ Convert physical state to corresponding reservoir hidden state,
    and reshape data into the format used in training.
    """
    # Concatenate variable tensors along the feature dimension,
    # which is assumed to be the last dim.
    batch_data_concat = concat_variables_along_feature_dim(variables, batch_data)

    # Convert data to latent representation if using an encoder
    if autoencoder is not None:
        logger.info("Using encoder to transform state data into latent space.")
        batch_data_concat = _encode_columns(batch_data_concat, autoencoder.encoder)

    # Divide into subdomains and flatten each subdomain by stacking
    # x/y/encoded-feature dims into a single subdomain-feature dimension.
    # Dimensions of a single subdomain's data become [time, subdomain-feature]
    X_subdomains_to_columns, Y_subdomains_to_columns = [], []
    for s in range(rank_divider.n_subdomains):
        X_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            batch_data_concat, subdomain_index=s, with_overlap=True,
        )
        X_subdomains_to_columns.append(stack_time_series_samples(X_subdomain_data))

        # Prediction does not include overlap
        Y_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            batch_data_concat, subdomain_index=s, with_overlap=False,
        )
        Y_subdomains_to_columns.append(stack_time_series_samples(Y_subdomain_data))

    # Concatentate subdomain data arrays along a new subdomain axis.
    # Dimensions are now [time, subdomain-feature, submdomain]
    X_reshaped = np.stack(X_subdomains_to_columns, axis=-1)
    Y_reshaped = np.stack(Y_subdomains_to_columns, axis=-1)

    return X_reshaped, Y_reshaped


def _get_reservoir_state_time_series(
    X: np.ndarray, input_noise: float, reservoir: Reservoir,
) -> np.ndarray:
    # Initialize hidden state
    if reservoir.state is None:
        reservoir.reset_state(input_shape=X[0].shape)

    # Increment and save the reservoir state after each timestep
    reservoir_state_time_series: List[Optional[np.ndarray]] = []
    for timestep_data in X:
        timestep_data = _add_input_noise(timestep_data, input_noise)
        reservoir.increment_state(timestep_data)
        reservoir_state_time_series.append(reservoir.state)
    return np.array(reservoir_state_time_series)


def _fit_batch_over_subdomains(
    subdomain_regressors,
    reservoir_state_time_series,
    prediction_time_series,
    square_even_inputs,
):
    # X has dimensions [time, reservoir_state, subdomain]
    X_batch = reservoir_state_time_series[:-1]
    # Y has dimensions [time, subdomain-feature, subdomain] where feature dimension
    # has flattened (x, y, encoded-variable) coordinates
    Y_batch = prediction_time_series[1:]
    if square_even_inputs is True:
        X_batch = square_even_terms(X_batch, axis=1)

    # Fit a readout to each subdomain's column of training data
    for i, regressor in enumerate(subdomain_regressors):
        # Last dimension is subdomains
        X_subdomain = X_batch[..., i]
        Y_subdomain = Y_batch[..., i]
        regressor.batch_update(
            X_subdomain, Y_subdomain,
        )
