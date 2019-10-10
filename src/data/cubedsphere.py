"""Tools for working with cubedsphere data"""
import pandas as pd
import xarray as xr
import numpy as np


NUM_TILES = 6


def remove_duplicate_coords(ds):
    deduped_indices = {}
    for dim in ds.dims:
        _, i = np.unique(ds[dim], return_index=True)
        deduped_indices[dim] = i
    return ds.isel(deduped_indices)


# TODO(Spencer): write a test of this function
def read_tile(prefix, tile, num_subtiles=16):
    subtiles = range(num_subtiles)
    filenames = [f'{prefix}.tile{tile:d}.nc.{proc:04d}' for proc in subtiles]
    return xr.open_mfdataset(
        filenames,
        data_vars='minimal',
        combine='by_coords'
    )


# TODO(Spencer): write a test of this function
def open_cubed_sphere(prefix: str, **kwargs):
    """Open cubed-sphere data

    Args:
        prefix: the beginning part of the filename before the `.tile1.nc.0001`
          part
    """
    tile_index = pd.Index(range(1, NUM_TILES + 1), name='tiles')
    tiles = [read_tile(prefix, tile, **kwargs) for tile in tile_index]
    combined = xr.concat(tiles, dim=tile_index)
    return remove_duplicate_coords(combined)
