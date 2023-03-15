# flake8: noqa

import functools
import os
import random
from typing import Optional, Tuple
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs
import numpy as np
import sklearn.preprocessing
import pickle

GRID = catalog["grid/c48"].read()

C48_I_TRAIN_END = 11688
C384_I_TRAIN_END = 2920


if __name__ == "__main__":
    c384_real_all: xr.Dataset = (
        xr.open_zarr("./fine-combined.zarr/").rename({"grid_xt": "x", "grid_yt": "y"})
    )
    precip = c384_real_all["PRATEsfc"].isel(time=slice(0, None, 9)).values.flatten()
    n_bins = 100
    vmin = 0.0
    vmax = 100
    v1 = 10 ** (np.log10(vmax) / n_bins)
    bins = np.concatenate([[vmin], np.logspace(np.log10(v1), np.log10(vmax), n_bins)])

    hist, _ = np.histogram(precip, bins=bins, density=True,)

    fig, ax = plt.subplots(1, 1, figsize=(5, 1 + 2.5),)
    ax.step(
        bins[:-1], hist, where="post", alpha=0.5,
    )
    ax.set_yscale("log")
    ax.set_title(f"C384 PRATEsfc")
    plt.tight_layout()
    fig.savefig("hist.png", dpi=100)
