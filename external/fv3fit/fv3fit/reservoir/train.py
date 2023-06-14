import logging
from joblib import Parallel, delayed
from fv3fit.reservoir.readout import BatchLinearRegressor
import numpy as np
import tensorflow as tf
from typing import Optional, Mapping, Tuple, List, Iterable, Union, Sequence
from .. import Predictor
from .utils import square_even_terms
from .autoencoder import Autoencoder, build_concat_and_scale_only_autoencoder
from .._shared import register_training_function
from ._reshaping import concat_inputs_along_subdomain_features
from . import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
    ReservoirComputingReadout,
)
from .readout import combine_readouts
from .domain import RankDivider, assure_same_dims
from ._reshaping import stack_samples

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _stack_array_preserving_last_dim(data):
    original_z_dim = data.shape[-1]
    reshaped = tf.reshape(data, shape=(-1, original_z_dim))
    return reshaped


def _encode_columns(data: Sequence[tf.Tensor], encoder: tf.keras.Model,) -> np.ndarray:
    # reduce a sequnence of N x M x Vi dim data over i variables
    # to a single N x M x Z dim array, where Vi is original number of features
    # (usually vertical levels) of each variable and Z << V is a smaller number
    # of latent dimensions
    original_sample_shape = data[0].shape[:-1]
    reshaped = [_stack_array_preserving_last_dim(var) for var in data]
    encoded_reshaped = encoder.predict(reshaped)
    return encoded_reshaped.reshape(*original_sample_shape, -1)


def _add_input_noise(arr: np.ndarray, stddev: float) -> np.ndarray:
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


def _get_ordered_X(X_mapping, variables):
    ordered_tensors = [X_mapping[v] for v in variables]
    return assure_same_dims(ordered_tensors)


@register_training_function("reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:

    sample_batch = next(iter(train_batches))
    sample_X = _get_ordered_X(sample_batch, hyperparameters.input_variables)

    if hyperparameters.autoencoder_path is not None:
        autoencoder = Autoencoder.load(hyperparameters.autoencoder_path)
    else:
        sample_X_stacked = [
            _stack_array_preserving_last_dim(arr).numpy() for arr in sample_X
        ]
        autoencoder = build_concat_and_scale_only_autoencoder(
            variables=hyperparameters.input_variables, X=sample_X_stacked
        )

    subdomain_config = hyperparameters.subdomain

    # sample_X[0] is the first data variable, shape elements 1:-1 are the x,y shape
    rank_extent = [*sample_X[0].shape[1:-1], autoencoder.n_latent_dims]
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
        time_series_with_overlap, time_series_without_overlap = _process_batch_Xy_data(
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
        hybrid_time_series: Optional[np.ndarray]
        if hyperparameters.hybrid_variables is not None:
            hybrid_time_series = _process_batch_hybrid_data(
                hybrid_variables=hyperparameters.hybrid_variables,
                batch_data=batch_data,
                rank_divider=rank_divider,
            )
        else:
            hybrid_time_series = None

        readout_input, readout_output = _construct_readout_inputs_outputs(
            reservoir_state_time_series,
            time_series_without_overlap,
            hyperparameters.square_half_hidden_state,
            hybrid_time_series=hybrid_time_series,
        )

        if b >= hyperparameters.n_batches_burn:
            logger.info(f"Fitting on batch {b+1}")
            _fit_batch_over_subdomains(
                X_batch=readout_input,
                Y_batch=readout_output,
                subdomain_regressors=subdomain_regressors,
                n_jobs=hyperparameters.n_jobs,
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

    model: Union[ReservoirComputingModel, HybridReservoirComputingModel]
    if hyperparameters.hybrid_variables is None:
        model = ReservoirComputingModel(
            input_variables=hyperparameters.input_variables,
            output_variables=hyperparameters.input_variables,
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=hyperparameters.square_half_hidden_state,
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
    else:
        model = HybridReservoirComputingModel(
            input_variables=hyperparameters.input_variables,
            output_variables=hyperparameters.input_variables,
            hybrid_variables=hyperparameters.hybrid_variables,
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=hyperparameters.square_half_hidden_state,
            rank_divider=rank_divider,
            autoencoder=autoencoder,
        )
    return model


def _process_batch_Xy_data(
    variables: Iterable[str],
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankDivider,
    autoencoder: Autoencoder,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Convert physical state to corresponding reservoir hidden state,
    and reshape data into the format used in training.
    """
    batch_X = _get_ordered_X(batch_data, variables)

    # Concatenate features, normalize and optionally convert data
    # to latent representation
    batch_data_encoded = _encode_columns(batch_X, autoencoder.encoder)
    # Divide into subdomains and flatten each subdomain by stacking
    # x/y/encoded-feature dims into a single subdomain-feature dimension.
    # Dimensions of a single subdomain's data become [time, subdomain-feature]
    X_subdomains_to_columns, Y_subdomains_to_columns = [], []
    for s in range(rank_divider.n_subdomains):
        X_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            batch_data_encoded,
            subdomain_index=s,
            with_overlap=True,
            data_has_time_dim=True,
        )
        X_subdomains_to_columns.append(
            stack_samples(X_subdomain_data, data_has_time_dim=True)
        )

        # Prediction does not include overlap
        Y_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            batch_data_encoded,
            subdomain_index=s,
            with_overlap=False,
            data_has_time_dim=True,
        )
        Y_subdomains_to_columns.append(
            stack_samples(Y_subdomain_data, data_has_time_dim=True)
        )

    # Concatentate subdomain data arrays along a new subdomain axis.
    # Dimensions are now [time, subdomain-feature, subdomain]
    X_reshaped = np.stack(X_subdomains_to_columns, axis=-1)
    Y_reshaped = np.stack(Y_subdomains_to_columns, axis=-1)

    return X_reshaped, Y_reshaped


def _process_batch_hybrid_data(
    hybrid_variables: Iterable[str],
    batch_data: Mapping[str, tf.Tensor],
    rank_divider: RankDivider,
) -> np.ndarray:
    # Similar to _process_batch_Xy_data for separating subdomains and
    # reshaping data, but does not transform the data into latent space
    batch_hybrid = _get_ordered_X(batch_data, hybrid_variables)
    batch_hybrid_combined_inputs = np.concatenate(batch_hybrid, axis=-1)
    hybrid_subdomains_to_columns = []
    for s in range(rank_divider.n_subdomains):
        hybrid_subdomain_data = rank_divider.get_subdomain_tensor_slice(
            batch_hybrid_combined_inputs,
            subdomain_index=s,
            with_overlap=True,
            data_has_time_dim=True,
        )
        hybrid_subdomains_to_columns.append(
            stack_samples(hybrid_subdomain_data, data_has_time_dim=True)
        )
    return np.stack(hybrid_subdomains_to_columns, axis=-1)


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


def _fit_batch(X_batch, Y_batch, subdomain_index, regressor):
    # Last dimension is subdomains
    X_subdomain = X_batch[..., subdomain_index]
    Y_subdomain = Y_batch[..., subdomain_index]
    regressor.batch_update(
        X_subdomain, Y_subdomain,
    )


def _construct_readout_inputs_outputs(
    reservoir_state_time_series,
    prediction_time_series,
    square_even_inputs,
    hybrid_time_series=None,
):
    # X has dimensions [time, reservoir_state, subdomain]
    X_batch = reservoir_state_time_series[:-1]
    if square_even_inputs is True:
        X_batch = square_even_terms(X_batch, axis=1)
    if hybrid_time_series is not None:
        X_batch = concat_inputs_along_subdomain_features(
            X_batch, hybrid_time_series[:-1]
        )
    # Y has dimensions [time, subdomain-feature, subdomain] where feature dimension
    # has flattened (x, y, encoded-feature) coordinates
    Y_batch = prediction_time_series[1:]
    return X_batch, Y_batch


def _fit_batch_over_subdomains(
    X_batch, Y_batch, subdomain_regressors, n_jobs,
):

    # Fit a readout to each subdomain's column of training data
    Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_fit_batch)(X_batch, Y_batch, i, regressor)
        for i, regressor in enumerate(subdomain_regressors)
    )
