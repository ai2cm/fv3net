"""Tools for working with cubedsphere data"""
import xarray as xr
from itertools import product
from toolz import groupby
import numpy as np


#TODO write a test for this method
def combine_subtiles(tiles):
    """Combine subtiles of a cubed-sphere dataset

    In v.12 of xarray, combined_by_coords behaves differently with DataArrays
    and Datasets. It was broadcasting all the variables to a common set of
    dimensions, dramatically increasing the size of the dataset.

    """
    data_vars = list(tiles[0])
    output_vars = []
    for key in data_vars:
        # the to_dataset is needed to avoid a strange error
        tiles_for_var = [tile[key].to_dataset() for tile in tiles]
        combined = xr.combine_by_coords(tiles_for_var)
        output_vars.append(combined)
    return xr.merge(output_vars)


def file_names(prefix, num_tiles=6, num_subtiles=16):

    tiles = list(range(1, num_tiles + 1))
    subtiles = list(range(num_subtiles))

    for (tile, proc) in product(tiles, subtiles):
        filename = prefix + f'.tile{tile:d}.nc.{proc:04d}'
        yield tile, proc, filename

#TODO test this
def remove_duplicate_coords(ds):
    deduped_indices = {}
    for dim in ds.dims:
        _, i = np.unique(ds[dim], return_index=True)
        deduped_indices[dim] = i
    return ds.isel(deduped_indices)


def open_cubed_sphere(prefix: str, **kwargs):
    """Open cubed-sphere data

    Args:
        prefix: the beginning part of the filename before the `.tile1.nc.0001`
          part
    """
    files = file_names(prefix, **kwargs)
    datasets = ((tile, xr.open_dataset(path, chunks={}))
                for tile, proc, path in files)
    tiles = groupby(lambda x: x[0], datasets)
    tiles = {tile: [data for tile, data in values]
             for tile, values in tiles.items()}

    combined_tiles = [combine_subtiles(datasets).assign_coords(tiles=tile)
                      for tile, datasets in tiles.items()]

    data = xr.concat(combined_tiles, dim='tiles').sortby('tiles')
    return remove_duplicate_coords(data)
