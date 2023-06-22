# flake8: noqa

import random
from typing import Mapping
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import numpy as np
import sklearn.preprocessing

GRID = catalog["grid/c48"].read()


def plot_hist(data: Mapping[str, np.ndarray]):
    fig, ax = plt.subplots(len(data), 2, figsize=(10, 1 + 2.5 * len(data)))
    for i, (name, values) in enumerate(data.items()):
        ax[i, 0].hist(
            values.flatten(), bins=100, alpha=0.5, label=name, histtype="step"
        )
        ax[i, 1].hist(
            values.flatten(), bins=100, alpha=0.5, label=name, histtype="step",
        )
        ax[i, 1].set_yscale("log")
        ax[i, 0].legend(loc="upper left")
        ax[i, 1].legend(loc="upper left")
        ax[i, 0].set_title(name)
        ax[i, 1].set_title(f"{name} (log y-axis)")
    plt.tight_layout()
    fig.savefig(f"data_histogram.png", dpi=100)


if __name__ == "__main__":
    random.seed(0)
    cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
        # "gs://vcm-ml-experiments/cyclegan/2023-01-20/cyclegan_c48_to_c384-prec-h500-w512-2e-4"  # prec/h500, 512-width, epoch 50
        "gs://vcm-ml-experiments/cyclegan/2023-01-22/cyclegan_c48_to_c384-prec-h500-w512-2e-4-epoch50"  # prec/h500, 512-width, epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-211721-5ddf5ef3-epoch_040/"  # reduced-vars, 512-width
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-174516-bfffae02-epoch_025/"  # h500-only
        # "gs://vcm-ml-experiments/cyclegan/2023-01-19/cyclegan_c48_to_c384-h500only-kernel4-2e-4"  # h500-only, epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230119-171215-64882f40-epoch_060/"  # precip-only
        # "gs://vcm-ml-experiments/cyclegan/2023-01-19/cyclegan_c48_to_c384-preconly-kernel4-2e-4"  # precip-only epoch 100
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230122-163054-99bf1aed-epoch_066/"  # precip-only, 2e-5 +100 epochs
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-173808-a5fd4151-epoch_051/"  # 2e-4, kernel4
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-183240-b191e306-epoch_048/"  # 2e-4, kernel3
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-003753-f8afe719-epoch_085/"  # log, 2e-5
        # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230118-003758-1324b685-epoch_035/"  # log, 2e-4
        # "gs://vcm-ml-experiments/cyclegan/2023-01-11/cyclegan_c48_to_c384-trial-0/"
    ).to(DEVICE)

    c384_real_all: xr.Dataset = (
        xr.open_zarr("./fine-0K.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )
    c384: xr.Dataset = c384_real_all.isel(time=slice(2920, 2930))
    # transformer = sklearn.preprocessing.PowerTransformer(method="yeo-johnson")
    # transformer = sklearn.preprocessing.PowerTransformer(method="box-cox")
    transformer = sklearn.preprocessing.QuantileTransformer(
        output_distribution="normal"
    )
    h500_norm = transformer.fit_transform(c384["h500"].values.reshape(-1, 1))
    precip_norm = transformer.fit_transform(
        c384["PRATEsfc"].values.reshape(-1, 1) + 1.0
    )
    plot_hist(
        {
            "h500": c384["h500"].values,
            "h500_norm": h500_norm,
            "PRATEsfc": c384["PRATEsfc"].values,
            "PRATEsfc_norm": precip_norm,
        }
    )
