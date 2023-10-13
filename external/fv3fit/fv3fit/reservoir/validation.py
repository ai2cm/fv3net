import numpy as np
from typing import Union, Optional, Sequence, Hashable, Mapping
import xarray as xr
import tensorflow as tf
import wandb
import matplotlib.pyplot as plt
from toolz import curry

from fv3fit.reservoir.adapters import ReservoirAdapterType
from fv3fit.tensorboard import plot_to_image


def _run_one_step_predictions(synced_model, inputs):

    # run one steps
    predictions = []
    for i in range(len(inputs.time)):
        current_input = inputs.isel(time=i)
        predictions.append(synced_model.predict(current_input))

    return predictions


def _run_rollout_predictions(synced_model, inputs):

    # run one steps
    predictions = []
    previous_state = xr.Dataset()
    for i in range(len(inputs.time)):
        current_input = inputs.isel(time=i)
        current_input.update(previous_state)
        previous_state = synced_model.predict(current_input)
        predictions.append(previous_state)

    return predictions


def log_metrics(metrics: Mapping[Hashable, xr.Dataset]) -> None:
    for name, metric in metrics.items():
        for field, value in metric.items():
            wandb.log({f"{name}/{field}": value.values})


def log_metric_plots(plottable_metrics: Mapping[Hashable, xr.Dataset]) -> None:
    # TODO: I can used rotate in vcm.cubedsphere to rotate if I have tile number
    # just grab the origin from the tile spec
    for name, metric in plottable_metrics.items():
        for field, value in metric.items():
            fig = plt.figure(dpi=120)
            value.plot()
            wandb.log({f"{name}/{field}": wandb.Image(plot_to_image(fig))})
            plt.close(fig)


def log_tile_time_avgs(time_avg_fields: Mapping[Hashable, xr.Dataset]) -> None:
    for name, field in time_avg_fields.items():
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        for timeseries_source, values in field.items():
            values.plot(ax=ax, label=timeseries_source)
        wandb.log({f"timeseries/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)

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
    spatial_mean = _mean(dim=["x", "y"], mask=mask, area=area)

    # synchronize
    model.reset_state()
    for i in range(n_sync_steps):
        model.increment_state(inputs.isel(time=i))

    if model.model.reservoir.state is None:
        raise ValueError("Reservoir state is None after synchronization.")
    synced_state = model.model.reservoir.state.copy()

    post_sync_inputs = inputs.isel(time=slice(n_sync_steps, -1))
    targets = targets.isel(time=slice(n_sync_steps + 1, None))

    # for baseline comparison
    persistence = post_sync_inputs.copy().isel(z=0)
    persistence = persistence.assign_coords(time=targets.time)
    persistence_errors = (persistence - targets).compute()

    def _run_validation_experiment(_step_func, prefix):
        model.model.reservoir.set_state(synced_state)
        predictions = _step_func(model, post_sync_inputs)
        predictions_ds = xr.concat(predictions, dim="time")
        predictions_ds.assign_coords(time=targets.time)

        errors = (predictions_ds - targets).compute()

        metrics = _calculate_scores(errors, persistence_errors, mean_func=global_mean)
        metrics = {f"{prefix}_{key}": value for key, value in metrics.items()}

        spatial_metrics = _calculate_scores(
            errors, persistence_errors, mean_func=temporal_mean
        )
        spatial_metrics = {
            f"{prefix}_spatial_{key}": value for key, value in spatial_metrics.items()
        }

        temporal_metrics = _calculate_scores(
            errors, persistence_errors, mean_func=spatial_mean
        )
        temporal_metrics = {
            f"{prefix}_temporal_{key}": value for key, value in temporal_metrics.items()
        }

        return metrics, spatial_metrics, temporal_metrics, predictions_ds

    (
        metrics,
        spatial_metrics,
        temporal_metrics,
        one_step_predictions,
    ) = _run_validation_experiment(_run_one_step_predictions, "one_step")
    (
        _metrics,
        _spatial_metrics,
        _temporal_metrics,
        rollout_predictions,
    ) = _run_validation_experiment(_run_rollout_predictions, "rollout")

    metrics.update(_metrics)
    spatial_metrics.update(_spatial_metrics)
    temporal_metrics.update(_temporal_metrics)

    metrics["combined_score"] = metrics["one_step_rmse"] + metrics["rollout_rmse"]

    field_tile_avgs = {}
    for field, value in one_step_predictions.items():
        field_tile_avgs[field] = xr.Dataset(
            {
                "one_step": spatial_mean(value).compute(),
                "rollout": spatial_mean(rollout_predictions[field]).compute(),
                "target": spatial_mean(targets[field]).compute(),
            }
        )

    print(field_tile_avgs)

    if wandb.run is not None:
        log_metrics(metrics)
        log_metric_plots(spatial_metrics)
        log_tile_time_avgs(field_tile_avgs)


@curry
def _mean(data, dim, mask=None, area=None):
    if mask is not None:
        data = data.where(mask)

    if area is not None:
        data = data.weighted(area)

    return data.mean(dim=dim).compute()


def _calculate_scores(errors, baseline_errors, mean_func):
    bias = mean_func(errors)
    mse = mean_func(errors ** 2)
    rmse = np.sqrt(mse)
    mae = mean_func(np.abs(errors))
    baseline_rmse = np.sqrt(mean_func(baseline_errors ** 2))
    skill = 1 - rmse / baseline_rmse

    return {
        "bias": bias,
        "mse": mse,
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "mae": mae,
        "skill": skill,
    }
