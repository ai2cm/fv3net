import xarray as xr
import numpy as np
from typing import TypeVar, Union, Tuple, Hashable, Any, Callable
from pathlib import Path

topo_folder = Path('./topography')
topo_folder.mkdir(exist_ok = True, parents = True)

topo384 = xr.open_zarr('gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/pire_atmos_static_coarse.zarr')

wts = xr.open_zarr('gs://vcm-ml-raw-flexible-retention/2021-07-19-PIRE/C3072-to-C384-res-diagnostics/grid_spec_coarse.zarr')

T_DataArray_or_Dataset = TypeVar("T_DataArray_or_Dataset", xr.DataArray, xr.Dataset)
CoordFunc = Callable[[Any, Union[int, Tuple[int]]], Any]

def coarsen_coords_coord_func(
    coordinate: np.ndarray, axis: Union[int, Tuple[int]] = -1
) -> np.ndarray:
    """xarray coarsen coord_func version of coarsen_coords.

    Note that xarray requires an axis argument for this to work, but it is not
    used by this function.  To coarsen dimension coordinates, xarray reshapes
    the 1D coordinate into a 2D array, with the rows representing groups of
    values to aggregate together in some way.  The length of the rows
    corresponds to the coarsening factor.  The value of the coordinate sampled
    every coarsening factor is just the first value in each row.

    Args:
        coordinate: 2D array of coordinate values
        axis: Axes to reduce along (not used)

    Returns:
        np.array
    """
    return (
        ((coordinate[:, 0] - 1) // coordinate.shape[1] + 1)
        .astype(int)
        .astype(np.float32)
    )

def _propagate_attrs(
    reference_obj: T_DataArray_or_Dataset, obj: T_DataArray_or_Dataset
) -> T_DataArray_or_Dataset:
    """Propagate attributes from the reference object to another.

    Args:
        reference_obj: input object
        obj: output object

    Returns:
        xr.DataArray or xr.Dataset
    """
    if isinstance(reference_obj, xr.Dataset):
        for variable in reference_obj:
            obj[variable].attrs = reference_obj[variable].attrs
    obj.attrs = reference_obj.attrs
    return obj


def weighted_block_average(
    obj: T_DataArray_or_Dataset,
    weights: xr.DataArray,
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_2",
    coord_func: Union[str, CoordFunc] = coarsen_coords_coord_func,
) -> T_DataArray_or_Dataset:
    """Coarsen a DataArray or Dataset through weighted block averaging.

    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.

    Args:
        obj: Input Dataset or DataArray.
        weights: Weights (e.g. area or pressure thickness).
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    numerator = (obj * weights).coarsen(coarsen_kwargs, coord_func=coord_func).sum()  # type: ignore # noqa
    denominator = weights.coarsen(coarsen_kwargs, coord_func=coord_func).sum()  # type: ignore # noqa
    result = numerator / denominator

    if isinstance(obj, xr.DataArray):
        result = result.rename(obj.name)

    return _propagate_attrs(obj, result)

topo48 = weighted_block_average(topo384, wts['area_coarse'], 8, 'grid_xt_coarse', 'grid_yt_coarse')

topo384 = topo384['zsurf_coarse'].values
topo48 = topo48['zsurf_coarse'].values

topo384_min, topo384_max, topo48_min, topo48_max = topo384.min(), topo384.max(), topo48.min(), topo48.max()

topo384_norm = (topo384 - topo384_min) / (topo384_max - topo384_min)
topo48_norm = (topo48 - topo48_min) / (topo48_max - topo48_min)

np.save('topography/topo384_norm.npy', topo384_norm)
np.save('topography/topo48_norm.npy', topo48_norm)
np.save('topography/topo384_min.npy', topo384_min)
np.save('topography/topo384_max.npy', topo384_max)
np.save('topography/topo48_min.npy', topo48_min)
np.save('topography/topo48_max.npy', topo48_max)