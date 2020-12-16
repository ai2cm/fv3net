import matplotlib.pyplot as plt

MATRIX_NAME = "jacobian_matrices.png"
LINE_NAME = "jacobian_lines.png"

def plot_jacobian(model, output_dir: str):
    jacobian_dict = model.jacobian()

    variables_3d = [var_ for var_ in mean if "z" in mean[var_].dims]
    variables_2d = [var_ for var_ in mean if "z" not in mean[var_].dims]

    fig, axs = plt.subplots(
        len(variables_3d),
        len(model.output_variables),
        figsize=(12, 12),
        constrained_layout=True,
    )
    for i, in_name in enumerate(variables_3d):
        for j, out_name in enumerate(model.y_packer.pack_names):
            pane = jacobian_dict[(in_name, out_name)]
            im = pane.plot.imshow(
                x=out_name, y=in_name, ax=axs[i, j], yincrease=False, xincrease=False
            )
            axs[i, j].set_ylabel(f"in ({in_name})")
            axs[i, j].set_xlabel(f"out ({out_name})")
            axs[i, j].xaxis.tick_top()
            axs[i, j].xaxis.set_label_position("top")
            plt.colorbar(im, ax=axs[i, j])

    fig.suptitle(url)
    fig.savefig(os.path.join(output_dir, MATRIX_NAME))

    fig, axs = plt.subplots(
        len(variables_2d),
        len(model.output_variables),
        figsize=(12, 12),
        constrained_layout=True,
    )
    for i, in_name in enumerate(variables_2d):
        for j, out_name in enumerate(model.y_packer.pack_names):
            pane = np.asarray(jacobian_dict[(in_name, out_name)])
            axs[i, j].plot(pane.ravel(), np.arange(pane.size))
            axs[i, j].set_xlabel(out_name)
            axs[i, j].set_title(f"change in {in_name}")
            axs[i, j].set_ylabel("vertical level")
    fig.savefig(os.path.join(output_dir, LINE_NAME))
