import xarray as xr
import os
from itertools import product
import dask.bag as db
from os.path import join
from toolz import curry
import numpy as np
import logging


def integerize(x):
    return np.round(x).astype(x.dtype)


#TODO: rename this to coarsen_by_area. It is not specific to surface data
def coarsen_sfc_data(data: xr.Dataset, factor: float, method="sum") -> xr.Dataset:
    """Coarsen a tile of surface data

    The data is weighted by the area.

    Args:
        data: surface data with area included
        factor: coarsening factor. For example, C3072 to C96 is a factor of 32.
    Returns:
        coarsened: coarse data without area

    """
    area = data["area"]
    data_no_area = data.drop("area")

    def coarsen_sum(x):
        coarsen_obj = x.coarsen(xaxis_1=factor, yaxis_1=factor)
        coarsened = getattr(coarsen_obj, method)()
        return coarsened

    coarsened = coarsen_sum(data_no_area * area) / coarsen_sum(area)
    coarse_coords = coarsen_coords(factor, data)

    # special hack for SLMASK (should be integer quantity)
    coarsened['slmsk'] = integerize(coarsened.slmsk)

    return coarsened.assign_coords(**coarse_coords).assign_attrs(
        {'coarsening_factor': factor, 'coarsening_method': method}
    )


#TODO: fix this name. it loads the data with area
def load_tile_proc(tile, subtile, path, grid_path):
    grid_spec_to_data_renaming = {"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}
    grid = xr.open_dataset(grid_path)

    area = grid.area.rename(grid_spec_to_data_renaming)
    pane = xr.open_dataset(path)

    data = xr.merge([pane, area])

    return data


def coarsen_coords(factor, tile):
    return {
        key: ((tile[key][::factor] - 1) // factor + 1).astype(int).astype(float)
        for key in ["xaxis_1", "yaxis_1"]
    }


#TODO use src.data.cubedsphere for this
def _combine_subtiles(tiles):
    combined = xr.concat(tiles, dim="io").sum("io")
    return combined.assign_attrs(tiles[0].attrs)


def combine_subtiles(args_list):
    tile, args_list = args_list
    subtiles = [arg[1] for arg in args_list]
    return tile, _combine_subtiles(subtiles).assign(tiles=tile)


def tile(args):
    return args[0][0]


@curry
def save_tile_to_disk(output_dir, args):
    tile, data = args
    output_name = f"sfc_data.tile{tile}.nc"
    output_path = join(output_dir, output_name)
    data.to_netcdf(output_path)
    return output_path


def coarsen_sfc_data_in_directory(files, **kwargs):
    """Process surface data in directory

    Args:
        files: list of (tile, subtile, sfc_data_path, grid_spec_path) tuples

    Returns:
        xarray of concatenated data
    """

    def process(args):
        logging.info(f"Coarsening {args}")
        data = load_tile_proc(*args)
        coarsened = coarsen_sfc_data(data, **kwargs)
        return args, coarsened

    bag = db.from_sequence(files)

    # res = bag.map(process)
    # res = res.compute(scheduler='single-threaded')

    procs = (
        bag
        .map(process)
        .groupby(tile)
        .map(combine_subtiles)
        .map(lambda x: x[1])
    )

    return xr.concat(procs.compute(scheduler='single-threaded'), dim='tiles')
