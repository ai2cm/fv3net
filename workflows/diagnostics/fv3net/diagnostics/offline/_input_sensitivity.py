import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import numpy as np
import xarray as xr

import fv3fit
from ._helpers import DATASET_DIM_NAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _stack_sample_data(ds: xr.Dataset) -> xr.Dataset:
    # for predictions, drop the 'target' values
    if "derivation" in ds.dims:
        ds = ds.sel({"derivation": "predict"})
    if "time" in ds.dims:
        ds = ds.isel(time=0).squeeze(drop=True)
    if DATASET_DIM_NAME in ds.dims:
        stack_dims = ["tile", "x", "y", DATASET_DIM_NAME]
    else:
        stack_dims = ["tile", "x", "y"]
    return ds.stack(sample=stack_dims).transpose("sample", ...)


def plot_input_sensitivity(model: fv3fit.Predictor, sample: xr.Dataset):
    base_model = model.base_model if isinstance(model, fv3fit.DerivedModel) else model
    stacked_sample = _stack_sample_data(sample)

    try:
        input_sensitivity: fv3fit.InputSensitivity = base_model.input_sensitivity(
            stacked_sample
        )
        if input_sensitivity.jacobians is not None:
            fig = _plot_jacobians(input_sensitivity.jacobians)
        elif input_sensitivity.rf_feature_importances is not None:
            fig = _plot_rf_feature_importance(input_sensitivity.rf_feature_importances)
        return fig

    except NotImplementedError:
        logger.info(
            f"Base model is {base_model.__class__.__name__}, "
            "which currently has no input_sensitivity method implemented."
        )
        return None


def _plot_rf_feature_importance(
    rf_input_sensitivity: fv3fit.RandomForestInputSensitivities,
):
    vector_features, scalar_features = {}, {}
    for name, feature in rf_input_sensitivity.items():
        info = {
            "indices": feature.indices,
            "mean_importances": feature.mean_importances,
            "std_importances": feature.std_importances,
        }
        if len(feature.indices) > 1:
            vector_features[name] = info
        else:
            scalar_features[name] = info
    n_panels = (
        len(vector_features) + 1 if len(scalar_features) > 0 else len(vector_features)
    )

    y_max = 1.1 * max(
        sum([info.mean_importances for info in rf_input_sensitivity.values()], [])
    )
    fig = plt.figure(figsize=(6 * n_panels, 4))

    for i, (name, info) in enumerate(vector_features.items()):
        ax = fig.add_subplot(1, n_panels, i + 1)

        ax.errorbar(
            info["indices"], info["mean_importances"], yerr=info["std_importances"],
        )
        ax.set_xlabel(f"{name} at feature dimension coordinate")
        ax.set_ylim(0, y_max)
    if len(scalar_features) > 0:
        xlabels = [k for k in scalar_features]
        y = sum([v["mean_importances"] for v in scalar_features.values()], [])
        yerr = sum([v["std_importances"] for v in scalar_features.values()], [])
        ax = fig.add_subplot(1, n_panels, n_panels)
        ax.bar(
            np.arange(len(scalar_features)),
            height=y,
            yerr=yerr,
            tick_label=xlabels,
            align="center",
        )
        ax.set_ylim(0, y_max)
    ax = fig.add_subplot(1, n_panels, 1)
    ax.set_ylabel("feature importance")
    plt.tight_layout()
    return fig


def _plot_jacobians(jacobians: fv3fit.JacobianInputSensitivity):
    num_outputs = len(jacobians)
    num_inputs = max([len(output) for output in jacobians.values()])
    fig, axes = plt.subplots(
        ncols=num_inputs,
        nrows=num_outputs,
        figsize=(4 * num_inputs, 4 * num_outputs),
        squeeze=False,
    )
    for i, (output_name, output) in enumerate(jacobians.items()):
        num_inputs = len(output)
        for j, (input_name, input) in enumerate(output.items()):
            ax = axes[i][j]
            if input.shape[-1] == 1:
                input = input.reshape(-1, 1)
                ax.plot(input)
                ax.set_ylim(-1, 1)
                ax.set_xlabel("input level")
                ax.set_ylabel("(dY / std(y)) / (dx / std(x))")
                ax.set_title(f"output: {output_name} \n input: {input_name}")
            else:
                im = ax.imshow(input, vmin=-1, vmax=1, aspect="auto", cmap="bwr")
                ax.set_xlabel("input level")
                ax.set_ylabel("output level")
                ax.invert_xaxis()
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax, orientation="vertical")
                ax.set_title(f"output: {output_name} \n input: {input_name}")
    fig.tight_layout()
    return fig
