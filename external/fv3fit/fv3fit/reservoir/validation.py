import numpy as np
<<<<<<< HEAD
from typing import Union, Optional, Sequence
import xarray as xr
import tensorflow as tf
import wandb
=======

>>>>>>> d8b43153e (Add metric function)
from toolz import curry
from fv3fit.reservoir.adapters import ReservoirAdapterType

from fv3fit.reservoir.utils import get_ordered_X
from fv3fit.reservoir import (
    ReservoirComputingModel,
    HybridReservoirComputingModel,
    HybridReservoirDatasetAdapter,
    ReservoirDatasetAdapter,
)

import logging

logger = logging.getLogger(__name__)

ReservoirModel = Union[ReservoirComputingModel, HybridReservoirComputingModel]
ReservoirAdapter = Union[HybridReservoirDatasetAdapter, ReservoirDatasetAdapter]


def _get_predictions_over_batch(
    model: ReservoirModel,
    states_with_overlap_time_series: Sequence[np.ndarray],
    hybrid_inputs_time_series: Optional[Sequence[np.ndarray]] = None,
):
    imperfect_prediction_time_series, prediction_time_series = [], []
    n_timesteps = states_with_overlap_time_series[0].shape[0]
    for t in range(n_timesteps):
        state = [
            variable_time_series[t]
            for variable_time_series in states_with_overlap_time_series
        ]
        model.increment_state(state)
        predict_kwargs = {}
        if hybrid_inputs_time_series is not None:
            predict_kwargs["hybrid_input"] = [
                variable_time_series
                for variable_time_series in hybrid_inputs_time_series[t]
            ]
            imperfect_prediction_time_series.append(predict_kwargs["hybrid_input"])
        prediction = model.predict(**predict_kwargs)  # type: ignore
        prediction_time_series.append(prediction)
    return prediction_time_series, imperfect_prediction_time_series


def _time_mean_dataset(variables, arr, label):
    ds = xr.Dataset()
    time_mean_error = np.mean(arr, axis=0)
    for v, var in enumerate(variables):
        ds[f"{label}_{var}"] = xr.DataArray(time_mean_error[v], dims=["x", "y", "z"])
    return ds


def _get_states_without_overlap(
    states_with_overlap_time_series: Sequence[np.ndarray], overlap: int
):
    states_without_overlap_time_series = []
    for var_time_series in states_with_overlap_time_series:
        # dims in array var_time_series are (t, x, y, z)
        states_without_overlap_time_series.append(
            var_time_series[:, overlap:-overlap, overlap:-overlap, :]
        )
    # dims (t, var, x, y, z)
    return np.stack(states_without_overlap_time_series, axis=1)


def validation_prediction(
    model: ReservoirModel, val_batches: tf.data.Dataset, n_synchronize: int,
):
    # Initialize hidden state
    model.reset_state()

    one_step_prediction_time_series = []
    one_step_imperfect_prediction_time_series = []
    target_time_series = []
    for batch_data in val_batches:
        states_with_overlap_time_series = get_ordered_X(
            batch_data, model.input_variables  # type: ignore
        )

        if isinstance(model, HybridReservoirComputingModel):
            hybrid_inputs_time_series = get_ordered_X(
                batch_data, model.hybrid_variables  # type: ignore
            )
            hybrid_inputs_time_series = _get_states_without_overlap(
                hybrid_inputs_time_series, overlap=model.rank_divider.overlap
            )
        else:
            hybrid_inputs_time_series = None

        batch_predictions, batch_imperfect_predictions = _get_predictions_over_batch(
            model, states_with_overlap_time_series, hybrid_inputs_time_series
        )

        one_step_prediction_time_series += batch_predictions
        one_step_imperfect_prediction_time_series += batch_imperfect_predictions
        target_time_series.append(
            _get_states_without_overlap(
                states_with_overlap_time_series, overlap=model.rank_divider.overlap
            )
        )
    target_time_series = np.concatenate(target_time_series, axis=0)[n_synchronize:]

    persistence = target_time_series[:-1]
    target = np.array(target_time_series[1:])

    # _get_predictions_over_batch predicts up to n_timesteps-1
    one_step_predictions = np.array(one_step_prediction_time_series)[n_synchronize:-1]
    time_means_to_calculate = {
        "time_mean_prediction": one_step_predictions,
        "time_mean_prediction_error": one_step_predictions - target,
        "time_mean_persistence_error": persistence - target,
        "time_mean_prediction_mse": (one_step_predictions - target) ** 2,
        "time_mean_persistence_mse": (persistence - target) ** 2,
    }

    if len(one_step_imperfect_prediction_time_series) > 0:
        one_step_imperfect_predictions = np.array(
            one_step_imperfect_prediction_time_series
        )[n_synchronize:-1]
        imperfect_prediction_time_means_to_calculate = {
            "time_mean_imperfect_prediction": one_step_imperfect_predictions,
            "time_mean_imperfect_prediction_error": one_step_imperfect_predictions
            - target,
            "time_mean_imperfect_prediction_mse": (
                one_step_imperfect_predictions - target
            )
            ** 2,
        }
        time_means_to_calculate.update(imperfect_prediction_time_means_to_calculate)

    diags_ = []
    for key, data in time_means_to_calculate.items():
        diags_.append(_time_mean_dataset(model.input_variables, data, key))

    return xr.merge(diags_)


UNITS = {
    "air_temperature": "K",
    "specific_humidity": "kg/kg",
    "air_pressure": "Pa",
    "eastward_wind": "m/s",
    "northward_wind": "m/s",
}


def log_rmse_z_plots(ds_val, variables):
    # Note: spatial mean is not area-averaged. If reporting metrics in publication
    # will need to change this.
    for var in variables:
        rmse = {}
        for comparison in ["persistence", "prediction", "imperfect_prediction"]:
            mse_key = f"time_mean_{comparison}_mse_{var}"
            if mse_key in ds_val:
                rmse[comparison] = np.sqrt(ds_val[mse_key].mean(["x", "y"])).values

        wandb.log(
            {
                f"val_rmse_zplot_{var}": wandb.plot.line_series(
                    xs=ds_val.z.values,
                    ys=list(rmse.values()),
                    keys=list(rmse.keys()),
                    title=f"validation RMSE {UNITS.get(var, '')}: {var}",
                    xname="model level",
                )
            }
        )


def log_rmse_scalar_metrics(ds_val, variables):
    scaled_errors_persistence, scaled_errors_imperfect = [], []
    for var in variables:
        log_data = {}
        for comparison in ["persistence", "prediction", "imperfect_prediction"]:
            mse_key = f"time_mean_{comparison}_mse_{var}"
            if mse_key in ds_val:
                log_data[mse_key] = ds_val[mse_key].mean().values
                log_data[mse_key.replace("mse", "rmse")] = np.sqrt(log_data[mse_key])
        # scaled errors are the average across variables of prediction/persistence
        # and prediction/imperfect prediction errors
        scaled_errors_persistence.append(
            log_data[f"time_mean_prediction_mse_{var}"]
            / log_data[f"time_mean_persistence_mse_{var}"]
        )
        try:
            scaled_errors_imperfect.append(
                log_data[f"time_mean_prediction_mse_{var}"]
                / log_data[f"time_mean_imperfect_prediction_mse_{var}"]
            )
        except (KeyError):
            pass

    log_data["val_rmse_prediction_vs_persistence_scaled_avg"] = np.sqrt(
        np.mean(scaled_errors_persistence)
    )
    try:
        log_data["val_rmse_prediction_vs_imperfect_scaled_avg"] = np.sqrt(
            np.mean(scaled_errors_imperfect)
        )
    except (KeyError):
        pass

    wandb.log(log_data)


def validate_model(
    model: ReservoirAdapterType,
    inputs: xr.Dataset,
    n_sync_steps: int,
    targets: xr.Dataset,
    mask=None,
    area=None,
):

    # want to do the index handling in this function
    if len(inputs.time) != len(targets.time):
        raise ValueError("Inputs and targets must have the same number of time steps.")

    global_mean = _mean(dim=targets.dims, mask=mask, area=area)
    temporal_mean = _mean(dim="time", mask=mask, area=area)

    # synchronize
    model.reset_state()
    for i in range(n_sync_steps):
        model.increment_state(inputs.isel(time=i))

    if model.model.reservoir.state is None:
        raise ValueError("Reservoir state is None after synchronization.")
    synced_state = model.model.reservoir.state.copy()

    post_sync_inputs = inputs.isel(time=slice(n_sync_steps, -1))
    targets = targets.isel(time=slice(n_sync_steps + 1, None))

    # run one steps
    predictions = []
    for i in range(len(post_sync_inputs.time)):
        current_input = post_sync_inputs.isel(time=i)
        model.increment_state(current_input)
        predictions.append(model.predict(current_input))
    predictions_ds = xr.concat(predictions, dim="time")
    predictions_ds.assign_coords(time=targets.time)

    metrics = _calculate_scores(
        predictions_ds - targets, targets, mean_func=global_mean
    )
    metrics = {f"one_step_{key}": value for key, value in metrics.items()}

    spatial_metrics = _calculate_scores(
        predictions_ds - targets, targets, mean_func=temporal_mean
    )
    spatial_metrics = {
        f"one_step_spatial_{key}": value for key, value in spatial_metrics.items()
    }

    # Run rollout
    model.model.reservoir.set_state(synced_state)

    predictions = []
    output_for_auto_regressive = xr.Dataset()
    for i in range(len(post_sync_inputs.time)):
        current_input = post_sync_inputs.isel(time=i)
        current_input.update(output_for_auto_regressive)
        model.increment_state(current_input)
        output_for_auto_regressive = model.predict(current_input)
        predictions.append(output_for_auto_regressive)

    predictions_ds = xr.concat(predictions, dim="time")
    predictions_ds.assign_coords(time=targets.time)

    rollout_metrics = _calculate_scores(
        predictions_ds - targets, targets, mean_func=global_mean
    )
    metrics.update({f"rollout_{key}": value for key, value in rollout_metrics.items()})
    spatial_rollout_metrics = _calculate_scores(
        predictions_ds - targets, targets, mean_func=temporal_mean
    )
    spatial_metrics.update(
        {
            f"rollout_spatial_{key}": value
            for key, value in spatial_rollout_metrics.items()
        }
    )

    metrics["combined_score"] = metrics["one_step_rmse"] + metrics["rollout_rmse"]

    return metrics, spatial_metrics


@curry
def _mean(data, dim, mask=None, area=None):
    if mask is not None:
        data = data.where(mask)

    if area is not None:
        data = data.weighted(area)

    return data.mean(dim=dim).compute()


def _calculate_scores(errors, target, mean_func):
    bias = mean_func(errors)
    mse = mean_func(errors ** 2)
    rmse = np.sqrt(mse)
    mae = mean_func(np.abs(errors))
    rms = mean_func(target ** 2)
    skill = 1 - rms / rmse

    return {
        "bias": bias,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "skill": skill,
    }
