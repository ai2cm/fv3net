import logging
from joblib import Parallel, delayed
import fv3fit
from fv3fit.reservoir.readout import (
    BatchLinearRegressor,
    combine_readouts_from_subdomain_regressors,
)
import numpy as np
import tensorflow as tf
from typing import Optional, List, Union, cast, Mapping
from .. import Predictor
from .utils import (
    square_even_terms,
    process_batch_data,
    get_ordered_X,
    assure_txyz_dims,
    SynchronziationTracker,
    get_standard_normalizing_transformer,
)
from .transformers import TransformerGroup, Transformer
from .._shared import register_training_function
from . import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    Reservoir,
    ReservoirTrainingConfig,
)
from .adapters import ReservoirDatasetAdapter, HybridReservoirDatasetAdapter
from .domain2 import RankXYDivider


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
            transformers[variable_group] = cast(Transformer, fv3fit.load(path))

    # If input transformer not specified, always create a standard norm transform
    if "input" not in transformers:
        transformers["input"] = get_standard_normalizing_transformer(
            hyperparameters.input_variables, sample_batch
        )

    # If output transformer not specified and output_variables != input_variables,
    # create a separate standard norm transform
    if "output" not in transformers:
        if hyperparameters.output_variables != hyperparameters.input_variables:
            transformers["output"] = get_standard_normalizing_transformer(
                hyperparameters.output_variables, sample_batch
            )
        else:
            transformers["output"] = transformers["input"]

    # If hybrid variables transformer not specified, and hybrid variables are defined,
    # create a separate standard norm transform
    if "hybrid" not in transformers:
        if hyperparameters.hybrid_variables is not None:
            transformers["hybrid"] = get_standard_normalizing_transformer(
                hyperparameters.hybrid_variables, sample_batch
            )
        else:
            transformers["hybrid"] = transformers["input"]
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
    rank_divider = RankXYDivider(
        subdomain_layout=subdomain_config.layout,
        overlap=subdomain_config.overlap,
        overlap_rank_extent=rank_extent,
        z_feature_size=transformers.input.n_latent_dims,
    )

    if hyperparameters.mask_land is True:
        input_mask_array: Optional[
            np.ndarray
        ] = rank_divider.get_all_subdomains_with_flat_feature(
            np.where(assure_txyz_dims(sample_batch["land_sea_mask"])[0] == 1.0, 0, 1)
        )
    else:
        input_mask_array = None

    # First data dim is time, the rest of the elements of each
    # subdomain+halo are are flattened into feature dimension
    reservoir = Reservoir(
        hyperparameters=hyperparameters.reservoir_hyperparameters,
        input_size=rank_divider.flat_subdomain_len,
        input_mask_array=input_mask_array,
    )

    # One readout is trained per subdomain when iterating over batches,
    # and they are combined after training into a CombinedReadout
    subdomain_regressors = [
        BatchLinearRegressor(hyperparameters.readout_hyperparameters)
        for r in range(rank_divider.n_subdomains)
    ]
    sync_tracker = SynchronziationTracker(
        n_synchronize=hyperparameters.n_timesteps_synchronize
    )
    for b, batch_data in enumerate(train_batches):
        input_time_series = process_batch_data(
            variables=hyperparameters.input_variables,
            batch_data=batch_data,
            rank_divider=rank_divider,
            autoencoder=transformers.input,
            trim_halo=False,
        )
        # If the output variables differ from inputs, use the transformer specific
        # to the output set to transform the output data
        _output_rank_divider_with_overlap = rank_divider.get_new_zdim_rank_divider(
            z_feature_size=transformers.output.n_latent_dims
        )
        output_time_series = process_batch_data(
            variables=hyperparameters.output_variables,
            batch_data=batch_data,
            rank_divider=_output_rank_divider_with_overlap,
            autoencoder=transformers.output,
            trim_halo=True,
        )

        # reservoir increment occurs in this call, so always call this
        # function even if X, Y are not used for readout training.
        reservoir_state_time_series = _get_reservoir_state_time_series(
            input_time_series, hyperparameters.input_noise, reservoir
        )
        sync_tracker.count_synchronization_steps(len(reservoir_state_time_series))

        hybrid_time_series: Optional[np.ndarray]

        if hyperparameters.hybrid_variables is not None:
            _hybrid_rank_divider_with_overlap = rank_divider.get_new_zdim_rank_divider(
                z_feature_size=transformers.hybrid.n_latent_dims
            )
            hybrid_time_series = process_batch_data(
                variables=hyperparameters.hybrid_variables,
                batch_data=batch_data,
                rank_divider=_hybrid_rank_divider_with_overlap,
                autoencoder=transformers.hybrid,
                trim_halo=True,
            )
        else:
            hybrid_time_series = None

        readout_input, readout_output = _construct_readout_inputs_outputs(
            reservoir_state_time_series,
            output_time_series,
            hyperparameters.square_half_hidden_state,
            hybrid_time_series=hybrid_time_series,
        )
        if sync_tracker.completed_synchronization:
            readout_input = sync_tracker.trim_synchronization_samples_if_needed(
                readout_input
            )
            readout_output = sync_tracker.trim_synchronization_samples_if_needed(
                readout_output
            )
            logger.info(f"Fitting on batch {b+1}")
            readout_input = rank_divider.subdomains_to_leading_axis(
                readout_input, flat_feature=True
            )
            output_rank_divider = (
                _output_rank_divider_with_overlap.get_no_overlap_rank_divider()
            )
            readout_output = output_rank_divider.subdomains_to_leading_axis(
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
            transformers=transformers,
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
            rank_divider=rank_divider,  # type: ignore
            transformers=transformers,
        )
        return HybridReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )


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
