"""Tools for working with cubedsphere data"""
import logging
from os.path import join
from typing import Any, Callable, Hashable, List, Mapping, Tuple, Union

import dask
import numpy as np
import xarray as xr
from skimage.measure import block_reduce as skimage_block_reduce

NUM_TILES = 6
SUBTILE_FILE_PATTERN = "{prefix}.tile{tile:d}.nc.{subtile:04d}"
STAGGERED_DIMS = ["grid_x", "grid_y"]


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
            "Variable to shift to center must be centered on one horizontal axis and edge-valued on the other."
        )


# TODO: write a test for this method
def combine_subtiles(subtiles):
    """Combine subtiles of a cubed-sphere dataset
    In v.12 of xarray, with data_vars='all' by default, combined_by_coords
    broadcasts all the variables to a common set of dimensions, dramatically
    increasing the size of the dataset.  In some cases this can be avoided by
    using data_vars='minimal'; however it then fails for combining data
    variables that contain missing values and is also slow due to the fact that
    it does equality checks between variables it combines.

    To work around this issue, we combine the data variables of the dataset one
    at a time with the default data_vars='all', and then merge them back
    together.
    """
    sample_subtile = subtiles[0]
    output_vars = []
    for key in sample_subtile:
        # combine_by_coords currently does not work on DataArrays; see
        # https://github.com/pydata/xarray/issues/3248.
        subtiles_for_var = [subtile[key].to_dataset() for subtile in subtiles]
        combined = xr.combine_by_coords(subtiles_for_var)
        output_vars.append(combined)
    return xr.merge(output_vars)


def subtile_filenames(prefix, tile, pattern=SUBTILE_FILE_PATTERN, num_subtiles=16):
    for subtile in range(num_subtiles):
        yield pattern.format(prefix=prefix, tile=tile, subtile=subtile)


def all_filenames(prefix, pattern=SUBTILE_FILE_PATTERN, num_subtiles=16):
    filenames = []
    for tile in range(1, NUM_TILES + 1):
        filenames.extend(subtile_filenames(prefix, tile, pattern, num_subtiles))
    return filenames


def remove_duplicate_coords(ds):
    deduped_indices = {}
    for dim in ds.dims:
        _, i = np.unique(ds[dim], return_index=True)
        deduped_indices[dim] = i
    return ds.isel(deduped_indices)


# TODO: this expects prefix as a path prefix to the coarsened tiles
def open_cubed_sphere(prefix: str, **kwargs):
    """Open cubed-sphere data

     Args:
         prefix: the beginning part of the filename before the `.tile1.nc.0001`
           part
     """
    tiles = []
    for tile in range(1, NUM_TILES + 1):
        files = subtile_filenames(prefix, tile, **kwargs)
        subtiles = [xr.open_dataset(file, chunks={}) for file in files]
        combined = combine_subtiles(subtiles).assign_coords(tile=tile)
        tiles.append(remove_duplicate_coords(combined))
    return xr.concat(tiles, dim="tile")


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


def block_coarsen(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = "xaxis_1",
    y_dim: Hashable = "yaxis_1",
    method: str = "sum",
    coord_func: Union[
        str, Mapping[Hashable, Union[Callable, str]]
    ] = coarsen_coords_coord_func,
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by performing an operation over blocks.

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

    Returns:
        xr.Dataset or xr.DataArray.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    coarsen_object = obj.coarsen(coarsen_kwargs, coord_func=coord_func)
    result = getattr(coarsen_object, method)()

    return result


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


def save_tiles_separately(sfc_data, prefix, output_directory):
    for i in range(6):
        output_path = join(output_directory, f"{prefix}.tile{i+1}.nc")
        logging.info(f"saving data to {output_path}")
        sfc_data.isel(tile=i).to_netcdf(output_path)
