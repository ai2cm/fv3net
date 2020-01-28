"""Tools for working with cubedsphere data"""
from functools import partial
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple, Union

import dask
import dask.array as dask_array
import numpy as np
import scipy.stats
import xarray as xr
from skimage.measure import block_reduce as skimage_block_reduce

from .. import xarray_utils
from vcm.cubedsphere.constants import COORD_X_OUTER, COORD_Y_OUTER

NUM_TILES = 6
SUBTILE_FILE_PATTERN = "{prefix}.tile{tile:d}.nc.{subtile:04d}"
STAGGERED_DIMS = [COORD_X_OUTER, COORD_Y_OUTER]


def rename_centered_xy_coords(cell_centered_da):
    """
    Args:
        cell_centered_da: data array that got shifted from edges to cell centers
    Returns:
        same input array with dims renamed to corresponding cell center dims
    """
    for dim in STAGGERED_DIMS:
        if dim in cell_centered_da.dims:
            cell_centered_da[dim] = cell_centered_da[dim] - 1
            cell_centered_da = cell_centered_da.rename({dim: dim + "t"})
    return cell_centered_da


def shift_edge_var_to_center(edge_var: xr.DataArray):
    """
    Args:
        edge_var: variable that is defined on edges of grid, e.g. u, v

    Returns:
        data array with the original variable at cell center
    """
    for staggered_dim in [dim for dim in STAGGERED_DIMS if dim in edge_var.dims]:
        return rename_centered_xy_coords(
            0.5
            * (edge_var + edge_var.shift({staggered_dim: 1})).isel(
                {staggered_dim: slice(1, None)}
            )
        )
    else:
        raise ValueError(
            "Variable to shift to center must be centered on one horizontal axis and "
            "edge-valued on the other."
        )


def coarsen_coords(
    coarsening_factor: int,
    reference_subtile: Union[xr.Dataset, xr.DataArray],
    dims: List[Hashable],
) -> Mapping[Hashable, xr.DataArray]:
    """Coarsen subtile coordinates found on restart files or diagnostics.

    Args:
        coarsening_factor: Integer coarsening factor to use.
        reference_subtile: Reference Dataset or DataArray.
        dims: List of dimension names to coarsen coordinates for.

    Returns:
        Dictionary mapping dimension names to xr.DataArray objects.
    """
    result = {}
    for dim in dims:
        result[dim] = (
            ((reference_subtile[dim][::coarsening_factor] - 1) // coarsening_factor + 1)
            .astype(int)
            .astype(np.float32)
        )
        result[dim] = result[dim].assign_coords({dim: result[dim]})
    return result


def coarsen_coords_coord_func(coordinate: np.array, axis: Union[int, Tuple[int]] = -1):
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


def add_coordinates(
    reference_obj: Union[xr.Dataset, xr.DataArray],
    coarsened_obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    dims: List[Hashable],
) -> Union[xr.Dataset, xr.DataArray]:
    """Add coarsened subtile coordinates to the coarsened xarray data
    structure.

    Args:
        reference_obj: Reference Dataset or DataArray.
        coarsened_obj: Coarsened Dataset or DataArray.
        coarsening_factor: Integer coarsening factor used.
        dims: List of dimension names to coarsen coordinates for.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    coarsened_coords = coarsen_coords(coarsening_factor, reference_obj, dims)

    if isinstance(coarsened_obj, xr.DataArray):
        result = coarsened_obj.assign_coords(coarsened_coords).rename(
            coarsened_obj.name
        )
    else:
        result = coarsened_obj.assign_coords(coarsened_coords)
    return result


def weighted_block_average(
    obj: Union[xr.Dataset, xr.DataArray],
    weights: xr.DataArray,
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_2",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
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
    result = (obj * weights).coarsen(
        coarsen_kwargs, coord_func=coord_func
    ).sum() / weights.coarsen(coarsen_kwargs, coord_func=coord_func).sum()

    if isinstance(obj, xr.DataArray):
        return result.rename(obj.name)
    else:
        return result


def edge_weighted_block_average(
    obj: Union[xr.Dataset, xr.DataArray],
    spacing: xr.DataArray,
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    edge: str = "x",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset along a block edge.

    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.

    Args:
        obj: Input Dataset or DataArray.
        spacing: Spacing of grid cells in the coarsening dimension.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        edge: Grid cell side to coarse-grain along {'x', 'y'}.
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.DataArray.
    """
    if edge == "x":
        coarsen_dim = x_dim
        downsample_dim = y_dim
    elif edge == "y":
        coarsen_dim = y_dim
        downsample_dim = x_dim
    else:
        raise ValueError(f"'edge' most be either 'x' or 'y'; got {edge}.")

    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = (spacing * obj).coarsen(
        coarsen_kwargs, coord_func=coord_func
    ).sum() / spacing.coarsen(coarsen_kwargs, coord_func=coord_func).sum()
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    # Separate logic is needed to apply coord_func to the downsample dimension
    # coordinate (if it exists), because it is not included in the call to
    # coarsen.
    return _coarsen_downsample_coordinate(
        obj, result, downsample_dim, coarsening_factor, coord_func
    )


def _is_dict_like(value: Any) -> bool:
    """This is a function copied from xarray's internals for use in determining
    whether an object is dictionary like."""
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


def _coarsen_downsample_coordinate(
    reference_obj, coarsened_obj, dim, coarsening_factor, coord_func
):
    """Utility function for coarsening the downsampled dimension in the block
    edge coarsening methods.

    This is needed, because the block edge methods to not use xarray's coarsen
    method on the downsampled dimension, so we need to apply coord_func
    manually.
    """
    if dim in reference_obj.coords:
        if _is_dict_like(coord_func) and dim in coord_func:
            function = coord_func.get(dim, "mean")
        else:
            function = coord_func

        coarsened_obj[dim] = _reduce_via_coord_func(
            reference_obj, dim, coarsening_factor, function
        )
    return coarsened_obj


def _reduce_via_coord_func(reference_obj, dim, coarsening_factor, function):
    """Reduce a DataArray via a function in the form of a string or callable."""
    if isinstance(function, str):
        # Use boundary='pad' so that xarray can coarsen the coordinates even if
        # the dimension size is not a multiple of the coarsening factor; this
        # is needed to support arbitrary coord_func arguments on staggered grid
        # dimensions.
        coarsen_obj = reference_obj[dim].coarsen(
            {dim: coarsening_factor}, boundary="pad"
        )
        return getattr(coarsen_obj, function)()
    else:
        return _xarray_block_reduce_dataarray(
            reference_obj[dim].drop(dim), {dim: coarsening_factor}, function
        )


def _block_reduce_dataarray_coordinates(reference_da, da, block_sizes, coord_func):
    """Coarsen the coordinates of a DataArray.

    This function coarsens the coordinates of a reference DataArray and adds
    them to a coarsened DataArray following the specification in the coord_func
    argument.

    This is a private method intended for use only in
    _xarray_block_reduce_dataaray; it is meant to allow for block_reduce to
    enable the same coordinate transformation behavior as xarray's coarsen
    method.
    """
    if not _is_dict_like(coord_func):
        coord_func = {d: coord_func for d in da.dims}

    for dim, size in block_sizes.items():
        if dim in reference_da.coords:
            function = coord_func.get(dim, "mean")
            da[dim] = _reduce_via_coord_func(reference_da, dim, size, function)

    # For any dimension we do not reduce, restore the original coordinate if it
    # exists.
    for dim in reference_da.dims:
        if dim not in block_sizes and dim in reference_da.coords:
            da[dim] = reference_da[dim]

    return da


def _compute_block_reduced_dataarray_chunks(da, ordered_block_sizes):
    """Compute chunk sizes of the block reduced array."""
    # TODO: we could make the error checking more user-friendly by checking all
    # dimensions for compatible chunk sizes first and displaying all invalid
    # chunks / dimension combinations in the error message rather than just the
    # first occurence.
    new_chunks = []
    if da.chunks is not None:
        for axis, (dim, block_size) in enumerate(zip(da.dims, ordered_block_sizes)):
            dimension_chunks = np.array(da.chunks[axis])
            if not np.all(dimension_chunks % block_size == 0):
                raise ValueError(
                    f"All chunks along dimension {dim} of DataArray must be "
                    f"divisible by the block_size, {block_size}."
                )
            else:
                new_chunks.append(tuple(dimension_chunks // block_size))
    return new_chunks


def _skimage_block_reduce_wrapper(
    data,
    reduction_function=None,
    ordered_block_sizes=None,
    cval=np.nan,
    new_chunks=None,
):
    """A thin wrapper around skimage.measure.block_reduce for use only in
    xr.apply_ufunc in _xarray_block_reduce_dataarray."""
    if isinstance(data, dask.array.Array):
        return dask.array.map_blocks(
            skimage_block_reduce,
            data,
            ordered_block_sizes,
            reduction_function,
            cval,
            dtype=data.dtype,
            chunks=new_chunks,
        )
    else:
        return skimage_block_reduce(data, ordered_block_sizes, reduction_function, cval)


def _xarray_block_reduce_dataarray(
    da: xr.DataArray,
    block_sizes: Mapping[Hashable, int],
    reduction_function: Callable,
    cval: float = np.nan,
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> xr.DataArray:
    """An xarray and dask compatible block_reduce function designed for
    DataArrays.

    This is a wrapper around skimage.measure.block_reduce, which
    enables functionality similar to xarray's built-in coarsen method, but
    allows for arbitrary custom reduction functions.  As is default in xarray's
    coarsen method, coordinate values along the coarse-grained dimensions are
    coarsened using a mean aggregation method.  Flexibility in the reduction
    function comes at the cost of more restrictive chunking requirements for
    dask arrays.

    Args:
        da: Input DataArray.
        block_sizes: Dictionary of dimension names mapping to block sizes.
            If the DataArray is chunked, all chunks along a given dimension
            must be divisible by the block size along that dimension.
        reduction_function: Array reduction function which accepts a tuple of
            axes to reduce along.
        cval: Constant padding value if array is not perfectly divisible by the block
            size.
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.DataArray.

    Raises:
        ValueError: If chunks along a dimension are not divisible by the block
            size for that dimension.
    """
    # If none of the dimensions in block_sizes exist in the DataArray, return
    # the DataArray unchanged; this is consistent with xarray's coarsen method.
    reduction_dimensions = set(block_sizes.keys()).intersection(set(da.dims))
    if not reduction_dimensions:
        return da

    # skimage.measure.block_reduce requires an ordered tuple of block sizes
    # with a length corresponding to the rank of the array. Non-reduced
    # dimensions must be filled with a block size of 1.
    ordered_block_sizes = tuple(block_sizes.get(dim, 1) for dim in da.dims)
    new_chunks = _compute_block_reduced_dataarray_chunks(da, ordered_block_sizes)

    result = xr.apply_ufunc(
        _skimage_block_reduce_wrapper,
        da,
        input_core_dims=[da.dims],
        output_core_dims=[da.dims],
        dask="allowed",
        exclude_dims=set(da.dims),
        kwargs={
            "reduction_function": reduction_function,
            "ordered_block_sizes": ordered_block_sizes,
            "cval": cval,
            "new_chunks": tuple(new_chunks),
        },
    )

    return _block_reduce_dataarray_coordinates(da, result, block_sizes, coord_func)


def xarray_block_reduce(
    obj: Union[xr.Dataset, xr.DataArray],
    block_sizes: Mapping[Hashable, int],
    reduction_function: Callable,
    cval: float = np.nan,
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """A generic block reduce function for xarray data structures.

    This is a wrapper around skimage.measure.block_reduce, which
    enables functionality similar to xarray's built-in coarsen method, but
    allows for arbitrary custom reduction functions.  As is default in xarray's
    coarsen method, coordinate values along the coarse-grained dimensions are
    coarsened using a mean aggregation method.  Flexibility in the reduction
    function comes at the cost of more restrictive chunking requirements for
    dask arrays.

    This is needed for multiple reasons:
        1. xarray's implementation of coarsen does not support using dask
           arrays for some coarsening operations (e.g. median).
        2. xarray's implementation of coarsen supports a limited set of
           reduction functions; this implementation supports arbitrary
           functions which take an array as input and reduce along a tuple of
           dimensions (e.g. we could use it for a mode reduction).

    This function should only be used if one of those two reasons applies;
    otherwise use xarray's coarsen method.

    Args:
        obj: Input Dataset or DataArray.
        block_sizes: Dictionary mapping dimension names to integer block sizes.
        reduction_function: Array reduction function which accepts a tuple of
            axes to reduce along.
        cval: Constant padding value if array is not perfectly divisible by the
            block size.
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    if isinstance(obj, xr.Dataset):
        return obj.apply(
            _xarray_block_reduce_dataarray,
            args=(block_sizes, reduction_function),
            cval=cval,
            coord_func=coord_func,
        )
    else:
        return _xarray_block_reduce_dataarray(
            obj, block_sizes, reduction_function, cval=cval, coord_func=coord_func
        )


def horizontal_block_reduce(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    reduction_function: Callable,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """A generic horizontal block reduce function for xarray data structures.

    This is a convenience wrapper around block_reduce for applying coarsening
    over n x n patches of array elements.  It should only be used if a dask
    implementation of the reduction method has not been implemented (e.g. for
    median) or if a custom reduction method is used that is not implemented in
    xarray.  Otherwise, block_coarsen should be used.

    Args:
        obj: Input Dataset or DataArray.
        coarsening_factor: Integer coarsening factor to use.
        reduction_function: Array reduction function which accepts a tuple of
            axes to reduce along.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    block_sizes = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    return xarray_block_reduce(
        obj, block_sizes, reduction_function, coord_func=coord_func
    )


def block_median(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by taking the median over blocks.

    Mainly meant for coarse-graining sfc_data variables.

    Args:
        obj: Input Dataset or DataArray.
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
    return horizontal_block_reduce(
        obj,
        coarsening_factor,
        np.median,
        x_dim=x_dim,
        y_dim=y_dim,
        coord_func=coord_func,
    )


def block_edge_sum(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    edge: str = "x",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by summing along a block edge.

    Mainly meant for coarse-graining the dx and dy variables from the original
    grid_spec.

    Args:
        obj: Input Dataset or DataArray.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        edge: Grid cell side to coarse-grain along {'x', 'y'}.
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    if edge == "x":
        coarsen_dim = x_dim
        downsample_dim = y_dim
    elif edge == "y":
        coarsen_dim = y_dim
        downsample_dim = x_dim
    else:
        raise ValueError(f"'edge' most be either 'x' or 'y'; got {edge}.")

    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = obj.coarsen(coarsen_kwargs, coord_func=coord_func).sum()
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    # Separate logic is needed to apply coord_func to the downsample dimension
    # coordinate (if it exists), because it is not included in the call to
    # coarsen.
    return _coarsen_downsample_coordinate(
        obj, result, downsample_dim, coarsening_factor, coord_func
    )


def _mode(arr: np.array, axis: int = 0, nan_policy: str = "propagate") -> np.array:
    """A version of scipy.stats.mode that only returns a NumPy array with the
    mode values along the given axis."""
    result = scipy.stats.mode(arr, axis=axis, nan_policy=nan_policy).mode

    # Note that the result always has a length-one extra dimension in place of
    # the dimension that was reduced.  We would like to squeeze that out to be
    # consistent with NumPy reduction functions (like np.mean).
    return np.squeeze(result, axis)


def _ureduce(arr: np.array, func: Callable, **kwargs) -> np.array:
    """Heavily adapted from NumPy: https://github.com/numpy/numpy/blob/
    b83f10ef7ee766bf30ccfa563b6cc8f7fd38a4c8/numpy/lib/
    function_base.py#L3343-L3395

    This moves the axes to reduce along to the end of the array, reshapes the
    array so that these axes are combined into a single axis, and finally
    reduces the array along that combined axis.

    This is a simplified version of the NumPy version; to be robust, we may
    want to follow the NumPy version and make sure we handle all possible types
    of axis arguments (e.g. single integers and tuples; currently this only
    handles tuples).  For now this private function contains the minimum needed
    to implement a mode reduction function that can be applied over multiple
    axes at once.

    Args:
        arr : Input NumPy array.
        func : Function which takes an axis argument and reduces along a single
            axis.
        **kwargs : Keyword arguments to pass to the function.

    Returns:
        NumPy array.
    """
    arr = np.asanyarray(arr)
    axis = kwargs.get("axis", None)
    if axis is not None:
        if not isinstance(axis, tuple) or any(ax < 0 for ax in axis):
            raise ValueError(
                f"Our in-house version of _ureduce currently only supports "
                "a tuple of positive integers as an 'axis' argument; got {axis}."
            )
        nd = arr.ndim
        if len(axis) == 1:
            kwargs["axis"] = axis[0]
        else:
            keep = set(range(nd)) - set(axis)
            nkeep = len(keep)
            for i, s in enumerate(sorted(keep)):
                arr = arr.swapaxes(i, s)
            arr = arr.reshape(arr.shape[:nkeep] + (-1,))
            kwargs["axis"] = len(arr.shape) - 1
    return func(arr, **kwargs)


def _mode_reduce(
    arr: np.array, axis: Tuple[int] = (0,), nan_policy: str = "propagate"
) -> np.array:
    """A version of _mode that reduces over a tuple of axes."""
    return _ureduce(arr, _mode, axis=axis, nan_policy=nan_policy)


def _block_mode(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
    nan_policy: str = "propagate",
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by taking the mode over blocks.

    See also: scipy.stats.mode.

    Args:
        obj: Input Dataset or DataArray.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.
        nan_policy: Defines how to handle when input contains nan. 'propagate'
            returns nan, raise' throws an error, 'omit' performs the calculations
            ignoring nan values. Default is 'propagate'.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    reduction_function = partial(_mode_reduce, nan_policy=nan_policy)
    return horizontal_block_reduce(
        obj,
        coarsening_factor,
        reduction_function,
        x_dim=x_dim,
        y_dim=y_dim,
        coord_func=coord_func,
    )


CUSTOM_COARSENING_FUNCTIONS = {"median": block_median, "mode": _block_mode}


def block_coarsen(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    method: str = "sum",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
    func_kwargs: Optional[Dict] = None,
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by performing an operation over blocks.

    In addition to the operations supported through xarray's coarsen method,
    this function also supports dask-compatible 'median' and 'mode' reduction
    methods.

    Args:
        obj: Input Dataset or DataArray.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        method: Name of coarsening method to use.
        coord_func: function that is applied to the coordinates, or a
            mapping from coordinate name to function.  See `xarray's coarsen
            method for details
            <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.coarsen.html>`_.
        func_kwargs: Additional keyword arguments to pass to the reduction function.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    if func_kwargs is None:
        func_kwargs = {}

    if method in CUSTOM_COARSENING_FUNCTIONS:
        return CUSTOM_COARSENING_FUNCTIONS[method](
            obj,
            coarsening_factor,
            x_dim=x_dim,
            y_dim=y_dim,
            coord_func=coord_func,
            **func_kwargs,
        )
    else:
        coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
        coarsen_object = obj.coarsen(coarsen_kwargs, coord_func=coord_func)
        return getattr(coarsen_object, method)()


def _upsample_staggered_or_unstaggered(obj, upsampling_factor, dim):
    """If the dimension size is even, uniformly repeat all points along the
    dimension; if the dimension size is odd, repeat all the points by
    the given factor except for the last value along the dimension."""
    dim_size = obj.sizes[dim]
    is_staggered_dim = dim_size % 2 == 1
    if is_staggered_dim:
        # dask.array.repeat does not currently take array arguments for
        # repeats; we work around this by splitting the array into portions
        # where we upsample with a uniform factor and then concatenate them
        # back together.
        a = xarray_utils.repeat(
            obj.isel({dim: slice(None, -1)}), upsampling_factor, dim
        )
        b = xarray_utils.repeat(obj.isel({dim: slice(-1, None)}), 1, dim)
        if isinstance(obj, xr.Dataset):
            # Use data_vars="minimal" here to prevent data variables without
            # the repeating dimension from being broadcast along the repeating
            # dimension during the concat step.
            return xr.concat([a, b], dim=dim, data_vars="minimal")
        else:
            return xr.concat([a, b], dim=dim)
    else:
        return xarray_utils.repeat(obj, upsampling_factor, dim)


def block_upsample(
    obj: Union[xr.Dataset, xr.DataArray], upsampling_factor: int, dims: List[Hashable]
) -> Union[xr.Dataset, xr.DataArray]:
    """Upsample an object by repeating values n times in each
    horizontal dimension.

    If a dimension has an odd size, it is assumed to be represent the grid cell
    interfaces.  In that case, values of the array are repeated with the
    upsampling factor, except for values on the upper boundary, which are not
    be repeated at all.

    Note that this function drops the coordinates along the dimensions
    specified, as it is not always obvious how one should infer what
    the missing coordinates should be.  Coordinates are not needed for our
    application currently in VCM, so we will defer the complexity of addressing
    this question for the time being.

    Args:
        obj: Input Dataset or DataArray.
        factor: Integer upsampling factor to use.
        dims: List of dimension names to upsample along; could be unstaggered
            or staggered.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    for dim in dims:
        obj = _upsample_staggered_or_unstaggered(obj, upsampling_factor, dim)
    return obj


def block_upsample_like(
    da: xr.DataArray,
    reference_da: xr.DataArray,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
) -> xr.DataArray:
    """Upsample a DataArray and sync its chunk and coordinate properties with a
    reference DataArray.
    The purpose of this function is to upsample a coarsened DataArray back out
    to its original resolution.  In doing so it:
      - Ensures the chunk sizes of the upsampled DataArray match the chunk sizes of
        the original DataArray.  This is useful, because block_upsample often
        produces chunk sizes that are too small for good performance when
        applied to dask arrays.
      - Adds horizontal dimension coordinates that match the reference DataArray.

    As other block-related functions, this function assumes that the upsampling
    factor is the same in the x and y dimension.

    Args:
        da (xr.DataArray): input DataArray.
        reference_da (xr.DataArray): DataArray to which da will be upsampled.
        x_dim (Hashable, optional): name of x-dimension. Defaults to "xaxis_1"
        y_dim (Hashable, optional): name of y-dimension. Defaults to "yaxis_1"

    Returns:
        xr.DataArray: upsampled da
    """
    x_is_staggered_dim = da.sizes[x_dim] % 2 == 1
    if x_is_staggered_dim:
        upsampling_factor = (reference_da.sizes[x_dim] - 1) // (da.sizes[x_dim] - 1)
    else:
        upsampling_factor = reference_da.sizes[x_dim] // da.sizes[x_dim]
    result = block_upsample(da, upsampling_factor, [x_dim, y_dim])
    if isinstance(da.data, dask_array.Array):
        result = result.chunk(reference_da.transpose(*result.dims).chunks)
    return result.assign_coords(
        {x_dim: reference_da[x_dim], y_dim: reference_da[y_dim]}
    )
