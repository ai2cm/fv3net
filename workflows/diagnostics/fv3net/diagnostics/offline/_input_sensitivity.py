import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
import numpy as np
from typing import Tuple, Hashable, Iterable, Dict, Mapping
import xarray as xr

import fv3fit
from fv3fit.sklearn._random_forest import SklearnWrapper
from fv3fit.keras.jacobian import compute_jacobians, nondimensionalize_jacobians

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OutputSensitivity = Dict[str, np.ndarray]


def dataset_to_dict_input(ds):
    if "time" in ds.dims:
        ds = ds.isel(time=0).squeeze(drop=True)
    stacked = ds.stack(sample=["tile", "x", "y"]).transpose("sample", ...)
    data = {}
    for var in stacked:
        # for predictions, drop the 'target' values
        if "derivation" in stacked[var].dims:
            values = stacked[var].sel({"derivation": "predict"}).values
        else:
            values = stacked[var].values
        if len(stacked[var].dims) == 1:
            values = values.reshape(-1, 1)
        data[var] = values
    return data


def _count_features_2d(
    quantity_names: Iterable[Hashable], dataset: xr.Dataset, sample_dim_name: str
) -> Dict[Hashable, int]:
    """
    count features for (sample[, z]) arrays.
    Copied from fv3fit._shared.packer, as this logic is pretty robust.
    """
    for name in quantity_names:
        if len(dataset[name].dims) > 2:
            value = dataset[name]
            raise ValueError(
                "can only count 1D/2D (sample[, z]) "
                f"variables, recieved values for {name} with dimensions {value.dims}"
            )
    return_dict = {}
    for name in quantity_names:
        value = dataset[name]
        if len(value.dims) == 1 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif value.dims[0] != sample_dim_name:
            raise ValueError(
                f"cannot count values for {name} whose first dimension is not the "
                f"sample dimension ({sample_dim_name}), has dims {value.dims}"
            )
        else:
            return_dict[name] = value.shape[1]
    return return_dict


def _get_variable_indices(
    data: xr.Dataset, variables: Iterable[Hashable]
) -> Dict[Hashable, Tuple[int, int]]:
    if "time" in data.dims:
        data = data.isel(time=0).squeeze(drop=True)
    stacked = data.stack(sample=["tile", "x", "y"])
    variable_dims = _count_features_2d(
        variables, stacked.transpose("sample", ...), "sample"
    )
    start = 0
    variable_indices = {}
    for var, var_dim in variable_dims.items():
        variable_indices[var] = (start, start + var_dim)
        start += var_dim
    return variable_indices


def plot_input_sensitivity(model: fv3fit.Predictor, sample: xr.Dataset):
    base_model = model.base_model if isinstance(model, fv3fit.DerivedModel) else model

    try:
        data_dict = dataset_to_dict_input(sample)
        jacobians = compute_jacobians(
            base_model.get_dict_compatible_model(),  # type: ignore
            data_dict,
            base_model.input_variables,
        )
        # normalize factors so sensitivities are comparable but still
        # preserve level-relative magnitudes
        std_factors = {name: np.std(data, axis=0) for name, data in data_dict.items()}
        jacobians_std = nondimensionalize_jacobians(jacobians, std_factors)
        fig = _plot_jacobians(jacobians_std)
        return fig

    except AttributeError:
        try:
            input_feature_indices = _get_variable_indices(
                data=sample, variables=base_model.input_variables
            )
            fig = _plot_rf_feature_importance(
                input_feature_indices, base_model  # type: ignore
            )
            return fig

        except AttributeError:
            logger.info(
                f"Base model is {type(base_model).__name__}, "
                "which currently has no feature importance or Jacobian "
                "calculation implemented."
            )
            return None


def _plot_rf_feature_importance(
    input_feature_indices: Dict[Hashable, Tuple[int, int]],
    wrapped_model: SklearnWrapper,
):
    mean_importances = wrapped_model.mean_importances
    std_importances = wrapped_model.std_importances

    vector_features, scalar_features = {}, {}
    for var, var_indices in input_feature_indices.items():
        start, stop = var_indices
        if stop - start == 1:
            scalar_features[var] = var_indices
        else:
            vector_features[var] = var_indices

    n_panels = (
        len(vector_features) + 1 if len(scalar_features) > 0 else len(vector_features)
    )
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False,)
    axs = _subplot_vector_feature_importances(
        axs, vector_features, mean_importances, std_importances
    )
    if len(scalar_features) > 0:
        axs = _subplot_scalar_feature_importances(
            axs, scalar_features, mean_importances, std_importances
        )

    for ax in axs[0]:
        ax.set_ylim(0.0, max(mean_importances) * 1.1)
    axs[0][0].set_ylabel("feature importance")
    plt.tight_layout()
    return fig


def _subplot_vector_feature_importances(
    axs, vector_feature_indices, mean_importances, std_importances
):
    for i, (feature, indices) in enumerate(vector_feature_indices.items()):
        start, stop = indices
        dim_length = stop - start
        axs[0, i].errorbar(
            range(dim_length),
            mean_importances[start:stop],
            std_importances[start:stop],
        )
        axs[0, i].set_xlabel(f"{feature}, model level")
    return axs


def _subplot_scalar_feature_importances(
    axs, scalar_feature_indices, mean_importances, std_importances
):
    # Plot the scalar feature importances together in the last axes object
    scalar_features = list(scalar_feature_indices)

    scalar_feature_mean_importances, scalar_feature_std_importances = [], []
    for feature, indices in scalar_feature_indices.items():
        feature_index = indices[0]
        scalar_feature_mean_importances.append(mean_importances[feature_index])
        scalar_feature_std_importances.append(std_importances[feature_index])

    axs[0, -1].bar(
        range(len(scalar_features)),
        scalar_feature_mean_importances,
        yerr=scalar_feature_std_importances,
        tick_label=scalar_features,
    )
    return axs


def _plot_jacobians(jacobians: Mapping[str, OutputSensitivity]):
    num_outputs = len(jacobians)
    num_inputs = max([len(output) for output in jacobians.values()])
    fig, axes = plt.subplots(
        ncols=num_inputs, nrows=num_outputs, figsize=(4 * num_inputs, 4 * num_outputs)
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
