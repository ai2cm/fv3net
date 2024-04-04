from io import BytesIO
import logging
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from toolz import curry
from typing import Optional, Hashable, Mapping
import wandb
import xarray as xr


from fv3fit.reservoir.adapters import ReservoirAdapterType


UNITS = {
    "sst": "K",
}

logger = logging.getLogger(__name__)


def _run_one_step_predictions(synced_model, inputs, hybrid):

    # run one steps
    predictions = []
    for i in range(len(inputs.time)):
        synced_model.increment_state(inputs.isel(time=i))
        current_hybrid = hybrid.isel(time=i) if hybrid else {}
        predictions.append(synced_model.predict(current_hybrid))

    return predictions


def _get_slice(src_len, dst_len):
    """
    src_len: length of previous state dimension to be inserted into current state
    dst_len: length of current state dimension
    """
    if src_len == dst_len:
        sl = slice(None)
    elif src_len < dst_len:
        diff = dst_len - src_len
        overlap = diff // 2
        sl = slice(overlap, -overlap)
    else:
        raise ValueError("src_len must be <= dst_len")

    return sl


def _insert_tile_to_overlap(current: xr.DataArray, previous: xr.DataArray):
    # we can't grab the halos for offline rollouts because there is no prediction
    # for other tiles.  Instead just grab the original overlap, which is *very*
    # optimistic for forecasting...

    slices = []
    try:
        for src_len, dst_len in zip(previous.shape, current.shape):
            slices.append(_get_slice(src_len, dst_len))
    except ValueError:
        raise ValueError(
            f"Expected overlap for current state ({current.shape}) to be larger than "
            f"or equal to overlap for previous predicted state ({previous.shape})."
        )

    current = current.copy(deep=True)
    current.values[slices] = previous.values
    return current


def _insert_previous_state(current: xr.Dataset, previous: xr.Dataset):
    if "z" not in previous.dims:
        previous = previous.expand_dims(dim="z", axis=-1)

    for key, previous_field in previous.items():
        current_field = current[key]
        if current_field.shape != previous_field.shape:
            updated_field = _insert_tile_to_overlap(current_field, previous_field)
        else:
            updated_field = previous_field
        current[key] = updated_field

    return current


def _run_rollout_predictions(synced_model, inputs, hybrid):

    # run one steps
    predictions = []
    previous_state = xr.Dataset()
    for i in range(len(inputs.time)):
        current_input = inputs.isel(time=i)
        current_input = _insert_previous_state(current_input, previous_state)
        synced_model.increment_state(current_input)
        current_hybrid = hybrid.isel(time=i) if hybrid else {}
        previous_state = synced_model.predict(current_hybrid)
        predictions.append(previous_state)

    return predictions


def plot_to_image(figure):
    """Converts a matplotlib figure object to an image for use with wandb.Image"""
    # Save the figure to a bytes buffer
    buf = BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    return Image.open(buf)


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
            if "skill" in str(name):
                kwargs = {"vmin": -1, "vmax": 1, "cmap": "RdBu_r"}
            else:
                kwargs = {}
            value.plot(**kwargs)
            wandb.log({f"{name}/{field}": wandb.Image(plot_to_image(fig))})
            plt.close(fig)


def log_tile_time_avgs(time_avg_fields: Mapping[Hashable, xr.Dataset]) -> None:
    for name, field in time_avg_fields.items():
        fig = plt.figure(dpi=120)
        ax = plt.gca()
        for timeseries_source, values in field.items():
            values.plot(ax=ax, label=timeseries_source)
        plt.legend()
        plt.ylabel(f"{name} [{UNITS.get(str(name), 'unknown')}]")
        wandb.log({f"timeseries/{name}": wandb.Image(plot_to_image(fig))})
        plt.close(fig)


def validate_model(
    model: ReservoirAdapterType,
    reservoir_inputs: xr.Dataset,
    hybrid_inputs: Optional[xr.Dataset],
    n_sync_steps: int,
    targets: xr.Dataset,
    mask=None,
    area=None,
):

    # Enforce length checks because we want to handle all dimension adjustments
    # for validation within this function
    if len(reservoir_inputs.time) != len(targets.time):
        raise ValueError("Inputs and targets must have the same number of time steps.")
    if hybrid_inputs and len(hybrid_inputs.time) != len(targets.time):
        raise ValueError("Inputs and targets must have the same number of time steps.")

    global_mean = _mean(dim=targets.dims, mask=mask, area=area)
    temporal_mean = _mean(dim="time", mask=mask, area=area)
    spatial_mean = _mean(dim=["x", "y"], mask=mask, area=area)

    # synchronize
    model.reset_state()
    for i in range(n_sync_steps):
        model.increment_state(reservoir_inputs.isel(time=i))

    if model.model.reservoir.state is None:
        raise ValueError("Reservoir state is None after synchronization.")
    synced_state = model.model.reservoir.state.copy()

    post_sync_inputs = reservoir_inputs.isel(time=slice(n_sync_steps, -1))
    post_sync_hybrid = (
        hybrid_inputs.isel(time=slice(n_sync_steps, -1)) if hybrid_inputs else None
    )

    persistence = targets.isel(time=slice(n_sync_steps, -1))
    targets = targets.isel(time=slice(n_sync_steps + 1, None))

    if "time" in targets:
        persistence = persistence.drop("time").assign_coords(time=targets.time)
    persistence_errors = (persistence - targets).compute()

    def _run_validation_experiment(_step_func, prefix):
        model.model.reservoir.set_state(synced_state)
        predictions = _step_func(model, post_sync_inputs, post_sync_hybrid)
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

    if wandb.run is not None:

        log_metrics(metrics)
        log_metric_plots(spatial_metrics)
        log_tile_time_avgs(field_tile_avgs)

    return metrics, spatial_metrics, temporal_metrics, field_tile_avgs


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
