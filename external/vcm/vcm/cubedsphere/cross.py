import xarray as xr
import numpy as np

from typing import Hashable

from dataclasses import dataclass
from functools import singledispatch

SW = 3
NW = 0
NE = 1
SE = 2

__all__ = ["to_cross"]


@dataclass
class Tile:
    x: int
    y: int
    origin: int


TOPOLOGY = {
    0: Tile(0, 1, SW),
    1: Tile(1, 1, SW),
    2: Tile(1, 2, SW),
    3: Tile(2, 1, NW),
    4: Tile(3, 1, NW),
    5: Tile(1, 0, SE),
}


@singledispatch
def rotate(data, origin, dest_origin):
    return np.rot90(data, origin - dest_origin, axes=(-2, -1))


@rotate.register(xr.Variable)
def _rotate_variable(data, src, dest, dims=["grid_yt", "grid_xt"]):
    return xr.apply_ufunc(
        lambda x: rotate(x, src, dest),
        data,
        input_core_dims=[dims],
        output_core_dims=[dims],
        dask="allowed",
    )


@rotate.register(xr.DataArray)
def _rotate_data_array(data, src, dest, dims=None):
    variable = rotate(data.variable, src, dest, dims)
    coords = {}
    for key in data.coords:
        if set(data[key].dims) >= set(dims):
            coords[key] = rotate(data[key].variable, src, dest, dims)
    return xr.DataArray(variable, coords=coords)


def to_cross(
    data: xr.DataArray,
    tile: Hashable = "tile",
    x: Hashable = "grid_xt",
    y: Hashable = "grid_yt",
) -> xr.DataArray:
    """Combine tiles into a single 2D field resembling a "cross"

    Useful for quickly plotting maps or applying other 2D image processing
    techniques.
    
    See Also:

        Weyn J, Durran D, and Caruana, 2019
        https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001705

    """
    tiles = []
    dims = [y, x]

    data = data.drop_vars(dims + [tile], errors="ignore").transpose(..., tile, y, x)
    null = xr.full_like(data.isel({tile: 0}), np.nan)
    for j in range(3):
        row = []
        for i in range(4):
            for tile_num, spec in TOPOLOGY.items():
                if spec.y == j and spec.x == i:
                    arr = rotate(
                        data.isel({tile: tile_num}), spec.origin, SW, dims=dims
                    )
                    row.append(arr)
                    break
            else:
                arr = null
                row.append(arr)

        tiles.append(row)
    return xr.combine_nested(tiles, concat_dim=dims)
