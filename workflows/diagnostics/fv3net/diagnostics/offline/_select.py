import numpy as np
from typing import Sequence
import xarray as xr
import cftime

from vcm.select import meridional_ring
import vcm


def nearest_time_batch_index(
    time: cftime.DatetimeJulian, time_batches: Sequence[Sequence[cftime.DatetimeJulian]]
):
    min_index = 0
    min_distance = min(abs(np.array(time_batches[0]) - time))
    for index, time_batch in enumerate(time_batches):
        distance = min(abs(np.array(time_batch) - time))
        if distance < min_distance:
            min_index = index
    return min_index


def meridional_transect(ds: xr.Dataset):
    transect_coords = meridional_ring()
    return vcm.interpolate_unstructured(ds, transect_coords)


def plot_transect(
    data: xr.DataArray,
    xaxis: str = "lat",
    yaxis: str = "pressure",
    column_dim: str = "derivation",
    dataset_dim: str = "dataset",
):
    row_dim = dataset_dim if dataset_dim in data.dims else None
    num_datasets = len(data[dataset_dim]) if dataset_dim in data.dims else 1
    figsize = (10, 4 * num_datasets)
    facetgrid = data.plot(
        y=yaxis,
        x=xaxis,
        yincrease=False,
        col=column_dim,
        row=row_dim,
        figsize=figsize,
        robust=True,
    )
    facetgrid.set_ylabels("Pressure [Pa]")
    facetgrid.set_xlabels("Latitude [deg]")

    f = facetgrid.fig
    return f
