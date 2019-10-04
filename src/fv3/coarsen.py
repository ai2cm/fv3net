import xarray as xr
import os
from itertools import product
import dask.bag as db
from os.path import join
from toolz import curry
import numpy as np


sfc_data_pattern = "sfc_data.tile{tile}.nc.{proc:04d}"


def integerize(x):
    return np.round(x).astype(x.dtype)


def coarsen_c3072_to_c48(data, method="sum"):
    """Coarsen a tile of surface data from C3072 to C96

    The data is weighted by the area.

    Args:
        data: surface data with area included

    Returns:
        coarsened: coarse data without area

    """
    factor = int(3072 / 48)
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
        {'coarsening_factor': 32, 'coarsening_method': method}
    )


def load_tile_proc(tile, proc, prefix):
    grid_spec_to_data_renaming = {"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}
    grid_path = join(prefix, f"grid_spec.tile{tile}.nc.{proc:04d}")
    grid = xr.open_dataset(grid_path)

    area = grid.area.rename(grid_spec_to_data_renaming)
    path = join(prefix, sfc_data_pattern.format(tile=tile, proc=proc))
    pane = xr.open_dataset(path)

    data = xr.merge([pane, area])

    return data


def coarsen_coords(factor, tile):
    return {
        key: ((tile[key][::factor] - 1) // factor + 1).astype(int).astype(float)
        for key in ["xaxis_1", "yaxis_1"]
    }


def _combine_subtiles(tiles):
    combined = xr.concat(tiles, dim="io").sum("io")
    return combined.assign_attrs(tiles[0].attrs)


def combine_subtiles(args_list):
    tile, args_list = args_list
    tiles = [arg[1] for arg in args_list]
    return tile, _combine_subtiles(tiles)


def tile(args):
    return args[0][1]


@curry
def save_tile_to_disk(output_dir, args):
    tile, data = args
    output_name = f"sfc_data.tile{tile}.nc"
    output_path = join(output_dir, output_name)
    data.to_netcdf(output_path)
    return output_path


def coarsen_sfc_data_in_directory(
    input_directory,
    num_processors_per_tile=16,
    num_tiles=6,
    **kwargs,
):

    procs = list(range(num_processors_per_tile))
    tiles = list(range(1, num_tiles + 1))

    args_list = product(procs, tiles)

    def process(args):
        proc, tile = args
        data = load_tile_proc(tile, proc, input_directory)
        coarsened = coarsen_c3072_to_c48(data, **kwargs)
        return coarsened

    bag = db.from_sequence(args_list)
    coarse_tiles = bag.map(process)

    procs = (
        db.zip(bag, coarse_tiles)
        .groupby(tile)
        .map(combine_subtiles)
        .map(lambda x: x[1])
    )

    return xr.concat(procs.compute(), dim='tiles')
