import numpy as np
from typing import Sequence
import xarray as xr

from vcm.select import meridional_ring
import vcm


def snapshot(batches: Sequence[xr.Dataset], random_seed: int = 0):
    batch_index = np.random.randint(len(batches))
    batch = batches[batch_index]

    time_index = np.random.randint(len(batch.time))
    snapshot = batch.isel({"time": time_index})

    return snapshot


def meridional_transect(
    ds: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray,
):
    transect_coords = meridional_ring()
    outputs = xr.merge([ds, lat, lon])
    return vcm.regrid.interpolate_unstructured(outputs, transect_coords)
