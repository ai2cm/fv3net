from typing import Union
import metpy.interpolate
import numpy as np
import xarray as xr
from scipy.spatial import KDTree

import vcm.calc.thermo

# for use in regridding values to the same vertical grid [Pa]
PRESSURE_GRID = xr.DataArray(
    [
        300.0,
        500.0,
        700.0,
        1000.0,
        2000.0,
        3000.0,
        5000.0,
        7000.0,
        10000.0,
        12500.0,
        15000.0,
        17500.0,
        20000.0,
        22500.0,
        25000.0,
        30000.0,
        35000.0,
        40000.0,
        45000.0,
        50000.0,
        55000.0,
        60000.0,
        65000.0,
        70000.0,
        75000.0,
        77500.0,
        80000.0,
        82500.0,
        85000.0,
        87500.0,
        90000.0,
        92500.0,
        95000.0,
        97500.0,
        100000.0,
    ],
    dims="pressure",
)


def interpolate_to_pressure_levels(
    field: xr.DataArray,
    delp: xr.DataArray,
    levels: xr.DataArray = PRESSURE_GRID,
    dim: str = "pfull",
) -> xr.DataArray:
    """Regrid an atmospheric field to a fixed set of pressure levels

    Args:
        field: atmospheric quantity defined on hybrid vertical coordinates
        delp: pressure thickness of model layers in Pa. Must be broadcastable with
            ``da``
        dim: the vertical dimension name
        levels: 1D DataArray of output pressure levels

    Returns:
        the atmospheric quantity defined on ``pressure_levels``.
    """
    return interpolate_1d(
        levels, field, vcm.calc.thermo.pressure_at_midpoint_log(delp, dim=dim), dim=dim,
    )


def interpolate_1d(
    xp: xr.DataArray, x: xr.DataArray, field: xr.DataArray, dim: str,
) -> xr.DataArray:
    """Interpolates data with any shape over a specified axis.

    Wraps metpy.interpolate.interplolate_1d

    Args:
        xp: 1-D Dataarray of desired output levels.
        x: the original coordinate of ``field``. Must have the
            same dims of ``field``, and increasing along the ``original_dim``
            dimension.
        field: the quantity to be regridded
        dim: the dimension to interpolate over

    Returns:
        the quantity interpolated at the levels in ``output_grid``
        
    See Also:
        https://unidata.github.io/MetPy/latest/api/generated/metpy.interpolate.interpolate_1d.html

    """

    output_grid = np.asarray(xp)
    out_dim = list(xp.dims)[0]

    def _interpolate(x: np.ndarray, field: np.ndarray) -> np.ndarray:
        # axis=-1 gives a broadcast error in the current version of metpy
        axis = field.ndim - 1
        return metpy.interpolate.interpolate_1d(output_grid, x, field, axis=axis)

    output = xr.apply_ufunc(
        _interpolate,
        x,
        field,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[out_dim]],
        output_sizes={out_dim: len(output_grid)},
        dask="parallelized",
        output_dtypes=[field.dtype],
    )

    # make the array have the same order of dimensions as before
    dim_order = [dim if dim in field.dims else out_dim for dim in output.dims]
    return output.transpose(*dim_order).assign_coords({out_dim: output_grid})


def interpolate_nd(xp: xr.DataArray, x: xr.DataArray, field: xr.DataArray) -> xr.DataArray:
    """Interpolate data along a single dimension

    Args:
        xp: the desired output coordinates
        x: the collocation points of the input data. must share all
            dimensions except 1 of the xp.
        field: the field to be interpolated. Must share dimensions with x.

    Returns:
        interpolated: field interpolated along the single dimension NOT
            shared by x and xp.

    """
    return xp


def _coords_to_points(coords, order):
    return np.stack([coords[key] for key in order], axis=-1)


def interpolate_unstructured(
    data: Union[xr.DataArray, xr.Dataset], coords
) -> Union[xr.DataArray, xr.Dataset]:
    """Interpolate an unstructured dataset

    This is similar to the fancy indexing of xr.Dataset.interp, but it works
    with unstructured grids. Only nearest neighbors interpolation is supported for now.

    Args:
        data: data to interpolate
        coords: dictionary of dataarrays with single common dim, similar to the
            advanced indexing provided ``xr.DataArray.interp``. These can,
            but do not have to be actual coordinates of the Dataset, but they should
            be in a 1-to-1 map with the the dimensions of the data. For instance,
            one can use this function to find the height of an isotherm, provided
            that the temperature is monotonic with height.
    Returns:
        interpolated dataset with the coords from coords argument as coordinates.
    """
    dims_in_coords = set()
    for coord in coords:
        for dim in coords[coord].dims:
            dims_in_coords.add(dim)

    if len(dims_in_coords) != 1:
        raise ValueError(
            "The values of ``coords`` can only have one common shared "
            "dimension. The coords have these dimensions: "
            f"`{dims_in_coords}`"
        )

    dim_name = dims_in_coords.pop()

    spatial_dims = set()
    for key in coords:
        for dim in data[key].dims:
            spatial_dims.add(dim)
    spatial_dims = list(spatial_dims)

    stacked = data.stack({dim_name: spatial_dims})
    order = list(coords)
    input_points = _coords_to_points(stacked, order)
    output_points = _coords_to_points(coords, order)
    tree = KDTree(input_points)
    _, indices = tree.query(output_points)
    output = stacked.isel({dim_name: indices})
    output = output.drop(dim_name)
    return output.assign_coords(coords)
