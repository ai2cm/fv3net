# coding: utf-8
import fv3fit
import xarray as xr
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


def unpack_matrix(x_packer, y_packer, J):
    jacobian_dict = {}
    j = 0
    for in_name in x_packer.pack_names:
        i = 0
        for out_name in y_packer.pack_names:
            size_in = x_packer.feature_counts[in_name]
            size_out = y_packer.feature_counts[out_name]

            jacobian_dict[(in_name, out_name)] = xr.DataArray(
                J[i : i + size_out, j : j + size_in].numpy(), dims=[out_name, in_name]
            )
            i += size_out
        j += size_in

    return jacobian_dict


def jacobian(model, mean: xr.Dataset):
    mean_tf = tf.convert_to_tensor(model.X_packer.to_array(mean))
    with tf.GradientTape() as g:
        g.watch(mean_tf)
        y = model.model(mean_tf)

    J = g.jacobian(y, mean_tf)[0, :, 0, :]
    return unpack_matrix(model.X_packer, model.y_packer, J)


# unstable
# url = "gs://vcm-ml-experiments/2020-12-03-sensitivity-to-amount-of-input-data/neural-network-128-timesteps-seed-0/trained_model"

# stable
url = "gs://vcm-ml-experiments/2020-12-03-sensitivity-to-amount-of-input-data/neural-network-128-timesteps-1-epochs-seed-0/trained_model"


def plot_jacobian(model_url):

    model = fv3fit.load(model_url)
    nfeat = sum(model.X_packer.feature_counts.values())

    mean = model.X_packer.to_dataset(model.X_scaler.mean[np.newaxis, :])
    jacobian_dict = jacobian(model, mean)

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

    fig.suptitle(url)
    plt.show()


if __name__ == "__main__":
    plot_jacobian(url)
