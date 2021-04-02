import fsspec
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
import os
import fv3fit.keras._models
import logging


MATRIX_NAME = "jacobian_matrices.png"
LINE_NAME = "jacobian_lines.png"


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
    inputs_2d, outputs_2d = map(set, zip(*pairs_2d)) if pairs_2d else [], []
    inputs_3d, outputs_3d = map(set, zip(*pairs_3d)) if pairs_3d else [], []

    if pairs_3d:
        fig, axs = plt.subplots(
            len(pairs_3d), len(outputs_3d), figsize=(12, 12), squeeze=False
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
            len(inputs_2d), len(outputs_2d), figsize=(12, 12), squeeze=False
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
