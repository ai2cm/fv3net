import logging
from joblib import Parallel, delayed
import fv3fit
from fv3fit.reservoir.readout import BatchLinearRegressor
import numpy as np
import tensorflow as tf
from typing import Optional, List, Union, cast, Mapping
from .. import Predictor
from .utils import (
    square_even_terms,
    process_batch_Xy_data,
    get_ordered_X,
    get_standard_normalizing_transformer,
)
from .transformers import TransformerGroup
from .._shared import register_training_function
from ._reshaping import concat_inputs_along_subdomain_features
from . import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
    ReservoirComputingReadout,
)
from .adapters import ReservoirDatasetAdapter, HybridReservoirDatasetAdapter
from .readout import combine_readouts
from .domain import RankDivider
from ._reshaping import stack_array_preserving_last_dim
from fv3fit.reservoir.transformers import ReloadableTransfomer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _add_input_noise(arr: np.ndarray, stddev: float) -> np.ndarray:
    return arr + np.random.normal(loc=0, scale=stddev, size=arr.shape)


def _get_transformers(
    sample_batch: Mapping[str, tf.Tensor], hyperparameters: ReservoirTrainingConfig
) -> TransformerGroup:
    # Load transformers with specified paths
    transformers = {}
    for variable_group in ["input", "output", "hybrid"]:
        path = getattr(hyperparameters.transformers, variable_group, None)
        if path is not None:
            transformers[variable_group] = cast(ReloadableTransfomer, fv3fit.load(path))

    # If input transformer not specified, always create a standard norm transform
    if "input" not in transformers:
        transformers["input"] = get_standard_normalizing_transformer(
            hyperparameters.input_variables, sample_batch
        )

    # If output transformer not specified and output_variables != input_variables,
    # create a separate standard norm transform
    if "output" not in transformers and (
        hyperparameters.output_variables != hyperparameters.input_variables
    ):
        transformers["output"] = get_standard_normalizing_transformer(
            hyperparameters.output_variables, sample_batch
        )

    # If hybrid variables transformer not specified, and hybrid variables are defined,
    # create a separate standard norm transform
    if "hybrid" not in transformers and hyperparameters.hybrid_variables is not None:
        transformers["hybrid"] = get_standard_normalizing_transformer(
            hyperparameters.hybrid_variables, sample_batch
        )

    return TransformerGroup(**transformers)


@register_training_function("reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: tf.data.Dataset,
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:

    sample_batch = next(iter(train_batches))
    sample_X = get_ordered_X(sample_batch, hyperparameters.input_variables)

    transformers = _get_transformers(sample_batch, hyperparameters)
    subdomain_config = hyperparameters.subdomain

    # sample_X[0] is the first data variable, shape elements 1:-1 are the x,y shape
    rank_extent = sample_X[0].shape[1:-1]
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
        input_size=rank_divider.subdomain_size_with_overlap
        * transformers.input.n_latent_dims,
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
            autoencoder=transformers.input,
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
                autoencoder=transformers.hybrid,
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
        return ReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
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
        return HybridReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )


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
