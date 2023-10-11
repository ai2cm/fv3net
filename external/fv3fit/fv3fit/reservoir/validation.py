import xarray as xr
import numpy as np

from toolz import curry
from fv3fit.reservoir.adapters import ReservoirAdapterType


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

    errors = (predictions_ds - targets).compute()

    metrics = _calculate_scores(errors, targets, mean_func=global_mean)
    metrics = {f"one_step_{key}": value for key, value in metrics.items()}

    spatial_metrics = _calculate_scores(errors, targets, mean_func=temporal_mean)
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

    errors = (predictions_ds - targets).compute()
    rollout_metrics = _calculate_scores(errors, targets, mean_func=global_mean)
    metrics.update({f"rollout_{key}": value for key, value in rollout_metrics.items()})
    spatial_rollout_metrics = _calculate_scores(
        errors, targets, mean_func=temporal_mean
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
