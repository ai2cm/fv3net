import numpy as np
from typing import Sequence
import xarray as xr

from diagnostics_utils import meridonal_ring
import vcm


def snapshot(batches: Sequence[xr.Dataset], random_seed: int=0):
    batch_index = np.random.randint(len(batches))
    batch = batches[batch_index]
    
    time_index = np.random.randint(len(batch.time))
    snapshot = batch.isel({"time": time_index})

    return snapshot.squeeze().drop("time")


def meridional_transect(
        ds: xr.Dataset,
        variables: Sequence[str]):
    transect_coords = meridonal_ring()
    outputs = vcm.safe.get_variables(ds, variables)
    return vcm.regrid.interpolate_unstructured(outputs, transect_coords)

