import logging
from joblib import Parallel, delayed
import fv3fit
from fv3fit.reservoir.readout import (
    BatchLinearRegressor,
    combine_readouts_from_subdomain_regressors,
)
import numpy as np
import tensorflow as tf
from typing import Optional, List, Union
from .. import Predictor
from .utils import square_even_terms, process_batch_Xy_data, get_ordered_X
from .transformers.autoencoder import build_concat_and_scale_only_autoencoder
from .._shared import register_training_function
from . import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
)
from .domain2 import RankXYDivider
from ._reshaping import stack_array_preserving_last_dim
from fv3fit.reservoir.transformers import ReloadableTransfomer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_input_noise(arr: np.ndarray, stddev: float) -> np.ndarray:
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


@register_training_function("reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:

    sample_batch = next(iter(train_batches))
    sample_X = get_ordered_X(sample_batch, hyperparameters.input_variables)

    if hyperparameters.autoencoder_path is not None:
        autoencoder: ReloadableTransfomer = fv3fit.load(
            hyperparameters.autoencoder_path
        )  # type: ignore
    else:
        sample_X_stacked = [
            stack_array_preserving_last_dim(arr).numpy() for arr in sample_X
        ]
        autoencoder = build_concat_and_scale_only_autoencoder(
            variables=hyperparameters.input_variables, X=sample_X_stacked
        )

    subdomain_config = hyperparameters.subdomain

    # sample_X[0] is the first data variable, shape elements 1:-1 are the x,y shape
    rank_extent = sample_X[0].shape[1:-1]
    rank_divider = RankXYDivider(
        subdomain_layout=subdomain_config.layout,
        overlap=subdomain_config.overlap,
        overlap_rank_extent=rank_extent,
        z_feature_size=autoencoder.n_latent_dims,
    )
    no_overlap_divider = rank_divider.get_no_overlap_rank_divider()

    # First data dim is time, the rest of the elements of each
    # subdomain+halo are are flattened into feature dimension
    reservoir = Reservoir(
        hyperparameters=hyperparameters.reservoir_hyperparameters,
        input_size=rank_divider.flat_subdomain_len,
    )

    # One readout is trained per subdomain when iterating over batches,
    # and they are combined after training into a CombinedReadout
    subdomain_regressors = [
        BatchLinearRegressor(hyperparameters.readout_hyperparameters)
        for r in range(rank_divider.n_subdomains)
    ]
    for b, batch_data in enumerate(train_batches):
        time_series_with_overlap, time_series_without_overlap = process_batch_Xy_data(
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
            _, hybrid_time_series = process_batch_Xy_data(
                variables=hyperparameters.hybrid_variables,
                batch_data=batch_data,
                rank_divider=rank_divider,
                autoencoder=autoencoder,
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
            readout_input = rank_divider.subdomains_to_leading_axis(
                readout_input, flat_feature=True
            )
            readout_output = no_overlap_divider.subdomains_to_leading_axis(
                readout_output, flat_feature=True
            )
            jobs = [
                delayed(regressor.batch_update)(readout_input[i], readout_output[i])
                for i, regressor in enumerate(subdomain_regressors)
            ]
            Parallel(n_jobs=hyperparameters.n_jobs, backend="threading")(jobs)

    readout = combine_readouts_from_subdomain_regressors(subdomain_regressors)

    model: Union[ReservoirComputingModel, HybridReservoirComputingModel]

    if hyperparameters.hybrid_variables is None:
        model = ReservoirComputingModel(
            input_variables=hyperparameters.input_variables,
            output_variables=hyperparameters.input_variables,
            reservoir=reservoir,
            readout=readout,
            square_half_hidden_state=hyperparameters.square_half_hidden_state,
            rank_divider=rank_divider,  # type: ignore
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
            rank_divider=rank_divider,  # type: ignore
            autoencoder=autoencoder,
        )
    return model


def _get_reservoir_state_time_series(
    X: np.ndarray, input_noise: float, reservoir: Reservoir,
) -> np.ndarray:
    # X is [time, subdomain, feature]

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


def _construct_readout_inputs_outputs(
    reservoir_state_time_series,
    prediction_time_series,
    square_even_inputs,
    hybrid_time_series=None,
):
    # X has dimensions [time, subdomain, reservoir_state]
    # hybrid has dimension [time, subdomain, hybrid_feature]
    X_batch = reservoir_state_time_series[:-1]
    if square_even_inputs is True:
        X_batch = square_even_terms(X_batch, axis=-1)
    if hybrid_time_series is not None:
        X_batch = np.concatenate((X_batch, hybrid_time_series[:-1]), axis=-1)

    # Y has dimensions [time, subdomain, flat_subdomain_feature] where feature
    # dimension has flattened (x, y, encoded-feature) coordinates
    Y_batch = prediction_time_series[1:]
    return X_batch, Y_batch
