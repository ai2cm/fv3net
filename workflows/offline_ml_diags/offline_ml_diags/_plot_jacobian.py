import matplotlib.pyplot as plt
import numpy as np
import os
import fv3fit.keras._models
import logging

MATRIX_NAME = "jacobian_matrices.png"
LINE_NAME = "jacobian_lines.png"

def plot_jacobian(model: fv3fit.keras._models.DenseModel, output_dir: str):
    jacobian_dict = model.jacobian()

    inputs = {in_name for in_name, out_name in jacobian_dict.data_vars}
    outputs = {out_name for in_name, out_name in jacobian_dict.data_vars}
    variables_3d = [var_ for var_ in inputs if jacobian_dict.sizes[var_] >= 1]
    variables_2d = [var_ for var_ in inputs if jacobian_dict.sizes[var_] == 1]

    fig, axs = plt.subplots(
        len(variables_3d),
        len(outputs),
        figsize=(12, 12),
        constrained_layout=True,
    )

    for i, in_name in enumerate(variables_3d):
        for j, out_name in enumerate(outputs):
            logging.debug(f"{in_name}_{out_name}")
            pane = jacobian_dict[(in_name, out_name)]
            im = pane.rename(f"{in_name}_from_{out_name}").plot.imshow(
                x=out_name, y=in_name, ax=axs[i, j], yincrease=False, xincrease=False
            )
            axs[i, j].set_ylabel(f"in ({in_name})")
            axs[i, j].set_xlabel(f"out ({out_name})")
            axs[i, j].xaxis.tick_top()
            axs[i, j].xaxis.set_label_position("top")
            plt.colorbar(im, ax=axs[i, j])

    fig.savefig(os.path.join(output_dir, MATRIX_NAME))
    fig, axs = plt.subplots(
        len(variables_2d),
        len(outputs),
        figsize=(12, 12),
        constrained_layout=True,
    )
    for i, in_name in enumerate(variables_2d):
        for j, out_name in enumerate(outputs):
            pane = np.asarray(jacobian_dict[(in_name, out_name)])
            axs[i, j].plot(pane.ravel(), np.arange(pane.size))
            axs[i, j].set_xlabel(out_name)
            axs[i, j].set_title(f"change in {in_name}")
            axs[i, j].set_ylabel("vertical level")
    fig.savefig(os.path.join(output_dir, LINE_NAME))
