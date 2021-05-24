import fsspec
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Mapping
from fv3fit.sklearn import SklearnWrapper
import fv3fit.keras._models


MATRIX_NAME = "jacobian_matrices.png"
LINE_NAME = "jacobian_lines.png"
RF_FEATURE_IMPORTANCE_NAME = "rf_feature_importances.png"


def _separate_dimensions(jacobian_dict):
    pairs_2d, pairs_3d = [], []
    for (input, output) in jacobian_dict.data_vars:
        if jacobian_dict.sizes[input] == 1 or jacobian_dict.sizes[output] == 1:
            pairs_2d.append((input, output))
        elif jacobian_dict.sizes[input] > 1 and jacobian_dict.sizes[output] > 1:
            pairs_3d.append((input, output))
    return pairs_2d, pairs_3d


def plot_jacobian(model: fv3fit.keras._models.DenseModel, output_dir: str):
    jacobian_dict = model.jacobian()

    pairs_2d, pairs_3d = _separate_dimensions(jacobian_dict)
    inputs_2d, outputs_2d = (
        {in_name for in_name, out_name in pairs_2d},
        {out_name for in_name, out_name in pairs_2d},
    )
    inputs_3d, outputs_3d = (
        {in_name for in_name, out_name in pairs_3d},
        {out_name for in_name, out_name in pairs_3d},
    )

    if pairs_3d:
        fig, axs = plt.subplots(
            len(inputs_3d),
            len(outputs_3d),
            figsize=(4 * len(inputs_3d), 4 * len(outputs_3d)),
            squeeze=False,
        )
        for i, in_name in enumerate(inputs_3d):
            for j, out_name in enumerate(outputs_3d):
                logging.debug(f"{in_name}_{out_name}")
                pane = jacobian_dict[(in_name, out_name)]
                im = pane.rename(f"{out_name}_from_{in_name}").plot.imshow(
                    x=out_name,
                    y=in_name,
                    ax=axs[i, j],
                    yincrease=False,
                    xincrease=False,
                    add_colorbar=False,
                )
                axs[i, j].set_ylabel(f"in ({in_name})")
                axs[i, j].set_xlabel(f"out ({out_name})")
                axs[i, j].xaxis.tick_top()
                axs[i, j].xaxis.set_label_position("top")
                plt.colorbar(im, ax=axs[i, j])
        plt.tight_layout()
        with fsspec.open(os.path.join(output_dir, MATRIX_NAME), "wb") as f:
            fig.savefig(f)

    if len(pairs_2d) > 0:
        fig, axs = plt.subplots(
            len(inputs_2d),
            len(outputs_2d),
            figsize=(4 * len(inputs_2d), 4 * len(outputs_2d)),
            squeeze=False,
        )
        for i, in_name in enumerate(inputs_2d):
            for j, out_name in enumerate(outputs_2d):
                pane = np.asarray(jacobian_dict[(in_name, out_name)])
                axs[i, j].plot(pane.ravel(), np.arange(pane.size))
                axs[i, j].set_xlabel(out_name)
                axs[i, j].set_title(f"change in {in_name}")
                axs[i, j].set_ylabel("vertical level")
        plt.tight_layout()
        with fsspec.open(os.path.join(output_dir, LINE_NAME), "wb") as f:
            fig.savefig(f)


def plot_rf_feature_importance(
    input_feature_indices: Mapping[str, int],
    wrapped_model: SklearnWrapper,
    output_dir: str,
) -> None:

    importances = []
    for member in wrapped_model.model.regressors:
        importances.append(member.feature_importances_)
    mean_importances = np.array(importances).mean(axis=0)
    std_importances = np.array(importances).std(axis=0)

    n_vector_features = len(
        [
            var
            for var in input_feature_indices
            if (input_feature_indices[var][1] - input_feature_indices[var][0] > 1)
        ]
    )
    fig, axs = plt.subplots(
        1, n_vector_features + 1, figsize=(6 * n_vector_features, 4), squeeze=False,
    )
    axs = _subplot_vector_feature_importances(
        axs, input_feature_indices, mean_importances, std_importances
    )
    axs = _subplot_scalar_feature_importances(
        axs, input_feature_indices, mean_importances, std_importances
    )
    for ax in axs[0]:
        ax.set_ylim(0.0, max(mean_importances) * 1.1)
    axs[0][0].set_ylabel("feature importance")
    plt.tight_layout()

    with fsspec.open(os.path.join(output_dir, RF_FEATURE_IMPORTANCE_NAME), "wb") as f:
        fig.savefig(f)


def _subplot_vector_feature_importances(
    axs, variable_indices, mean_importances, std_importances
):
    vector_features = [
        var
        for var in variable_indices
        if (variable_indices[var][1] - variable_indices[var][0] > 1)
    ]
    for i, feature in enumerate(vector_features):
        dim_length = variable_indices[feature][1] - variable_indices[feature][0]
        axs[0, i].errorbar(
            range(dim_length),
            mean_importances[
                variable_indices[feature][0] : variable_indices[feature][1]
            ],
            std_importances[
                variable_indices[feature][0] : variable_indices[feature][1]
            ],
        )
        axs[0, i].set_xlabel(f"{feature}, model level")
    return axs


def _subplot_scalar_feature_importances(
    axs, variable_indices, mean_importances, std_importances
):
    # Plot the scalar feature importances together in the last axes object
    scalar_features = [
        var
        for var in variable_indices
        if (variable_indices[var][1] - variable_indices[var][0] == 1)
    ]

    scalar_feature_mean_importances = [
        mean_importances[variable_indices[var][0]] for var in scalar_features
    ]
    scalar_feature_std_importances = [
        std_importances[variable_indices[var][0]] for var in scalar_features
    ]
    axs[0, -1].bar(
        range(len(scalar_features)),
        scalar_feature_mean_importances,
        yerr=scalar_feature_std_importances,
        tick_label=["cos_z", "land_sea_mask", "phis"],
    )
    return axs
