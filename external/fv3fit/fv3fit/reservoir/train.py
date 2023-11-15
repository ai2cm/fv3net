import logging
from joblib import Parallel, delayed
import fv3fit
import numpy as np
import tensorflow as tf
from typing import Optional, List, Union, cast, Mapping, Sequence
import wandb

from fv3fit.reservoir.readout import (
    BatchLinearRegressor,
    combine_readouts_from_subdomain_regressors,
)
from .. import Predictor
from .utils import (
    square_even_terms,
    process_batch_data,
    process_validation_batch_data_to_dataset,
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
from .validation import validation_prediction, log_rmse_z_plots, log_rmse_scalar_metrics
from .validation import validate_model


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


def _expand_mask_zdim(mask: tf.Tensor, z_dim_len: int) -> tf.Tensor:
    if mask.shape[-1] != z_dim_len and mask.shape[-1] == 1:
        mask = mask * tf.ones(shape=(*mask.shape[:-1], z_dim_len))
    else:
        raise ValueError(
            f"Mask variable must have trailing dim of 1 or {z_dim_len}",
            f"but has len {mask.shape[-1]}.",
        )

    return mask


def _get_input_mask_array(
    mask_variable: str,
    sample_batch: Mapping[str, tf.Tensor],
    rank_divider: RankXYDivider,
    trim_halo: bool = False,
) -> np.ndarray:
    if mask_variable not in sample_batch:
        raise KeyError(
            f"'{mask_variable}' must be included in training data if "
            "the mask_variable is specified in training configuration."
        )
    mask = assure_txyz_dims(sample_batch[mask_variable])
    mask = mask * np.ones(
        rank_divider._rank_extent_all_features
    )  # broadcast feature dim

    if trim_halo:
        mask = rank_divider.trim_halo_from_rank_data(mask)
        rank_divider = rank_divider.get_no_overlap_rank_divider()

    mask = rank_divider.get_all_subdomains_with_flat_feature(mask[0])
    if set(np.unique(mask)) != {0, 1}:
        raise ValueError(
            f"Mask variable values in field {mask_variable} are not " "all in {0, 1}."
        )

    return mask


@register_training_function("reservoir", ReservoirTrainingConfig)
def train_reservoir_model(
    hyperparameters: ReservoirTrainingConfig,
    train_batches: Union[tf.data.Dataset, Sequence[tf.data.Dataset]],
    validation_batches: Optional[tf.data.Dataset],
) -> Predictor:
    train_batches_sequence = (
        train_batches if isinstance(train_batches, Sequence) else [train_batches]
    )
    sample_batch = next(iter(train_batches_sequence[0]))
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

    if hyperparameters.mask_variable is not None:
        input_mask_array: Optional[np.ndarray] = _get_input_mask_array(
            hyperparameters.mask_variable, sample_batch, rank_divider
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

    for train_batches in train_batches_sequence:
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
            if b == 0:
                reservoir.reset_state(input_shape=input_time_series[0].shape)
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
                _hybrid_rank_divider_w_overlap = rank_divider.get_new_zdim_rank_divider(
                    z_feature_size=transformers.hybrid.n_latent_dims
                )

                hybrid_time_series = process_batch_data(
                    variables=hyperparameters.hybrid_variables,
                    batch_data=batch_data,
                    rank_divider=_hybrid_rank_divider_w_overlap,
                    autoencoder=transformers.hybrid,
                    trim_halo=True,
                )

                if (
                    hyperparameters.mask_variable is not None
                    and hyperparameters.mask_readout
                ):
                    hybrid_input_mask_array = _get_input_mask_array(
                        hyperparameters.mask_variable,
                        batch_data,
                        _hybrid_rank_divider_w_overlap,
                        trim_halo=True,
                    )
                    hybrid_time_series = hybrid_time_series * hybrid_input_mask_array
                else:
                    hybrid_input_mask_array = None
            else:
                hybrid_time_series = None
                hybrid_input_mask_array = None

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
    adapter_model: Union[ReservoirDatasetAdapter, HybridReservoirDatasetAdapter]

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
        adapter_model = ReservoirDatasetAdapter(
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
            hybrid_input_mask=hybrid_input_mask_array,
        )
        adapter_model = HybridReservoirDatasetAdapter(
            model=model,
            input_variables=model.input_variables,
            output_variables=model.output_variables,
        )

    if wandb.run is not None and validation_batches is not None:
        if not hyperparameters.validate_sst_only:
            try:
                ds_val = validation_prediction(
                    model,
                    val_batches=validation_batches,
                    n_synchronize=hyperparameters.n_timesteps_synchronize,
                )
                log_rmse_z_plots(ds_val, model.output_variables)
                log_rmse_scalar_metrics(ds_val, model.output_variables)
            except Exception as e:
                logging.error("Error logging validation metrics to wandb", exc_info=e)
        else:
            data = next(iter(validation_batches))
            input_data = process_validation_batch_data_to_dataset(
                data, adapter_model.nonhybrid_input_variables
            )

            if adapter_model.is_hybrid:
                adapter_model = cast(HybridReservoirDatasetAdapter, adapter_model)
                hybrid_data = process_validation_batch_data_to_dataset(
                    data, adapter_model.hybrid_variables, trim_divider=rank_divider
                )
            else:
                hybrid_data = None

            output_vars = list(adapter_model.output_variables)
            if "mask_field" in data:
                output_vars.append("mask_field")
            if "area" in data:
                output_vars.append("area")

            target_data = process_validation_batch_data_to_dataset(
                data, output_vars, trim_divider=rank_divider
            ).squeeze()

            output_mask = target_data.isel(time=0).get("mask_field", None)
            area = target_data.isel(time=0).get("area", None)
            target_data = target_data.drop_vars(["mask_field", "area"], errors="ignore")

            validate_model(
                adapter_model,
                input_data,
                hybrid_data,
                hyperparameters.n_timesteps_synchronize,
                target_data,
                mask=output_mask,
                area=area,
            )

    return adapter_model


def _get_reservoir_state_time_series(
    X: np.ndarray, input_noise: float, reservoir: Reservoir,
) -> np.ndarray:
    # X is [time, subdomain, feature]

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
