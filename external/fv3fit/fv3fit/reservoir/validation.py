import xarray as xr
import numpy as np

from toolz import curry


def validate_model(model, inputs, n_sync_steps, targets, mask=None, area=None):

    # want to do the index handling in this function
    if len(inputs.time) != len(targets.time):
        raise ValueError("Inputs and targets must have the same number of time steps.")

    global_mean = _mean(dim=targets.dims, mask=mask, area=area)
    temporal_mean = _mean(dim="time", mask=mask, area=area)

    # synchronize
    model.reset_state()
    for i in range(n_sync_steps):
        model.increment(inputs.isel(time=i))

    post_sync_inputs = inputs.isel(time=slice(n_sync_steps, -1))
    targets = targets.isel(time=slice(n_sync_steps + 1, None))

    # run one steps
    predictions = []
    for i in range(len(post_sync_inputs.time)):
        current_input = post_sync_inputs.isel(time=i)
        model.increment(current_input)
        predictions.append(model.predict(current_input))
    predictions = xr.concat(predictions, dim="time")
    predictions.assign_coords(time=targets.time)

    metrics = _calculate_scores(predictions - targets, targets, mean_func=global_mean)
    metrics = {f"one_step_{key}": value for key, value in metrics.items()}

    spatial_metrics = _calculate_scores(
        predictions - targets, targets, mean_func=temporal_mean
    )
    spatial_metrics = {
        f"one_step_spatial_{key}": value for key, value in spatial_metrics.items()
    }

    # Run rollout
    model.reset_state()
    for i in range(n_sync_steps):
        model.increment(inputs.isel(time=i))

    predictions = []
    output_for_auto_regressive = {}
    for i in range(len(post_sync_inputs.time)):
        current_input = post_sync_inputs.isel(time=i)
        current_input.update(output_for_auto_regressive)
        model.increment(current_input)
        output_for_auto_regressive = model.predict(current_input)
        predictions.append(output_for_auto_regressive)

    predictions = xr.concat(predictions, dim="time")
    predictions.assign_coords(time=targets.time)

    rollout_metrics = _calculate_scores(
        predictions - targets, targets, mean_func=global_mean
    )
    metrics.update({f"rollout_{key}": value for key, value in rollout_metrics.items()})
    spatial_rollout_metrics = _calculate_scores(
        predictions - targets, targets, mean_func=temporal_mean
    )
    spatial_metrics.update(
        {
            f"rollout_spatial_{key}": value
            for key, value in spatial_rollout_metrics.items()
        }
    )

    return metrics, spatial_metrics


@curry
def _mean(data, dim, mask=None, area=None):
    if mask is not None:
        data = data.where(mask)

    if area is not None:
        data = data.weighted(area)

    return data.mean(dim=dim)


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
