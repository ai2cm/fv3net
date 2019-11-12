"""Tools for working with cubedsphere data"""
import dask
import numpy as np
import xarray as xr

from typing import Callable, Hashable, List, Mapping, Union

from skimage.measure import block_reduce as skimage_block_reduce


NUM_TILES = 6
SUBTILE_FILE_PATTERN = '{prefix}.tile{tile:d}.nc.{subtile:04d}'


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


def subtile_filenames(prefix, tile, pattern=SUBTILE_FILE_PATTERN,
                      num_subtiles=16):
    for subtile in range(num_subtiles):
        yield pattern.format(prefix=prefix, tile=tile, subtile=subtile)


def all_filenames(prefix, pattern=SUBTILE_FILE_PATTERN,
                  num_subtiles=16):
    filenames = []
    for tile in range(1, NUM_TILES + 1):
        filenames.extend(subtile_filenames(prefix, tile, pattern,
                                           num_subtiles)
        )
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
    return xr.concat(tiles, dim='tile')


def coarsen_subtile_coordinates(
    coarsening_factor: int,
    reference_subtile: Union[xr.Dataset, xr.DataArray],
    dims: List[Hashable]
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
            (reference_subtile[dim][::coarsening_factor] - 1)
            // coarsening_factor + 1
        ).astype(int).astype(np.float32)
        result[dim] = result[dim].assign_coords({dim: result[dim]})
    return result


def add_coarsened_subtile_coordinates(
    reference_obj: Union[xr.Dataset, xr.DataArray],
    coarsened_obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    dims: List[Hashable]
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
    coarsened_coordinates = coarsen_subtile_coordinates(
        coarsening_factor,
        reference_obj,
        dims
    )

    if isinstance(coarsened_obj, xr.DataArray):
        result = coarsened_obj.assign_coords(
            coarsened_coordinates
        ).rename(coarsened_obj.name)
    else:
        result = coarsened_obj.assign_coords(coarsened_coordinates)
    return result


def weighted_block_average(
    obj: Union[xr.Dataset, xr.DataArray],
    weights: xr.DataArray,
    coarsening_factor: int,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_2'
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset through weighted block averaging.

    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.

    Args:
        obj: Input DataArray or Dataset.
        weights: Weights (e.g. area or pressure thickness).
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').

    Returns:
        xr.Dataset or xr.DataArray.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    result = ((obj * weights).coarsen(coarsen_kwargs).sum() /
              weights.coarsen(coarsen_kwargs).sum())

    if isinstance(obj, xr.DataArray):
        return result.rename(obj.name)
    else:
        return result


def edge_weighted_block_average(
    obj: Union[xr.Dataset, xr.DataArray],
    spacing: xr.DataArray,
    coarsening_factor: int,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1',
    edge: str = 'x'
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset along a block edge.

    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.

    Args:
        obj: Input DataArray or Dataset.
        spacing: Spacing of grid cells in the coarsening dimension.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        edge: Grid cell side to coarse-grain along {'x', 'y'}.

    Returns:
        xr.DataArray.
    """
    if edge == 'x':
        coarsen_dim = x_dim
        downsample_dim = y_dim
    elif edge == 'y':
        coarsen_dim = y_dim
        downsample_dim = x_dim
    else:
        raise ValueError(f"'edge' most be either 'x' or 'y'; got {edge}.")

    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = (
        (spacing * obj).coarsen(coarsen_kwargs).sum() /
        spacing.coarsen(coarsen_kwargs).sum()
    )
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return result


def _block_reduce_dataarray(
    da: xr.DataArray,
    block_sizes: Mapping[Hashable, int],
    reduction_function: Callable
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
            must be divisible by the block size along that dimension.  All
            dimensions must be present in block_sizes (if no coarsening is to
            be done along a dimension, the block size should be specified as
            1).
        reduction_function: A function which reduces the array to a scalar.

    Returns:
        xr.DataArray.

    Raises:
        ValueError: If chunks along a dimension are not divisible by the block
            size for that dimension.
    """
    def func(data, ordered_block_sizes=None, new_chunks=None):
        if isinstance(data, dask.array.Array):
            return dask.array.map_blocks(
                skimage_block_reduce,
                data,
                ordered_block_sizes,
                reduction_function,
                dtype=data.dtype,
                chunks=new_chunks
            )
        else:
            return skimage_block_reduce(
                data,
                ordered_block_sizes,
                reduction_function
            )

    # TODO: we could make the error checking more user-friendly by checking all
    # dimensions for compatible chunk sizes first and displaying all invalid
    # chunks / dimension combinations in the error message rather than just the
    # first occurence.
    new_chunks = []
    if da.chunks is not None:
        for axis, dim in enumerate(da.dims):
            dimension_chunks = np.array(da.chunks[axis])
            if not np.all(dimension_chunks % block_sizes[dim] == 0):
                raise ValueError(
                    f'All chunks along dimension {dim} of DataArray must be '
                    f'divisible by the block_size, {block_sizes[dim]}.')
            else:
                new_chunks.append(tuple(dimension_chunks // block_sizes[dim]))

    ordered_block_sizes = tuple(block_sizes[dim] for dim in da.dims)

    result = xr.apply_ufunc(
        func,
        da,
        input_core_dims=[da.dims],
        output_core_dims=[da.dims],
        dask='allowed',
        exclude_dims=set(da.dims),
        kwargs={'ordered_block_sizes': ordered_block_sizes,
                'new_chunks': tuple(new_chunks)}
    )

    # Restore coordinates for dimensions that did not change size; for
    # coordinates that did change size, use a mean coarsen reduction (as is
    # default for xarray's coarsen method).
    for dim, size in block_sizes.items():
        if dim in da.coords:
            if size == 1:
                result[dim] = da[dim]
            else:
                result[dim] = da[dim].coarsen({dim: size}).mean()

    return result


def _horizontal_block_reduce_dataarray(
    da: xr.DataArray,
    coarsening_factor: int,
    reduction_function: Callable,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1'
) -> xr.DataArray:
    """A generic horizontal block reduce function for DataArrays.

    If neither x_dim or y_dim are in the DataArray, no reduction operation is
    performed.

    Args:
        da: Input DataArray.
        coarsening_factor: Integer coarsening factor to use.
        reduction_function: Function which reduces an array to a scalar.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').

    Returns:
        xr.DataArray.
    """
    if x_dim not in da.dims and y_dim not in da.dims:
        return da
    else:
        block_sizes = {}
        for dim in da.dims:
            if dim in [x_dim, y_dim]:
                block_sizes[dim] = coarsening_factor
            else:
                block_sizes[dim] = 1

        return _block_reduce_dataarray(da, block_sizes, reduction_function)


def horizontal_block_reduce(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    reduction_function: Callable,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1'
) -> Union[xr.Dataset, xr.DataArray]:
    """A generic horizontal block reduce function for xarray data structures.

    This is a wrapper around skimage.measure.block_reduce, which
    enables functionality similar to xarray's built-in coarsen method, but
    allows for arbitrary custom reduction functions.  As is default in xarray's
    coarsen method, coordinate values along the coarse-grained dimensions are
    coarsened using  mean aggregation method.  Flexibility in the reduction
    function comes at the cost of more restrictive chunking requirements for
    dask arrays.

    This is needed for multiple reasons:
        1. xarray's implementation of coarsen does not support using dask
           arrays for some coarsening operations (e.g. median).
        2. xarray's implementation of coarsen supports a limited set of
           reduction functions; this implementation supports arbitrary
           functions which take an array as input and reduce along a tuple of
           dimensions (e.g. we could use it for a mode reduction).

    Args:
        obj: Input DataArray or Dataset.
        coarsening_factor: Integer coarsening factor to use.
        reduction_function: Function which reduces an array to a scalar.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').

    Returns:
        xr.Dataset or xr.DataArray.
    """
    if isinstance(obj, xr.Dataset):
        return obj.apply(
            _horizontal_block_reduce_dataarray,
            args=(coarsening_factor, reduction_function),
            x_dim=x_dim, y_dim=y_dim
        )
    else:
        return _horizontal_block_reduce_dataarray(
            obj,
            coarsening_factor,
            reduction_function,
            x_dim=x_dim,
            y_dim=y_dim
        )


def block_median(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1'
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by taking the median over blocks.

    Mainly meant for coarse-graining sfc_data variables.

    Args:
        obj: Input DataArray or Dataset.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').

    Returns:
        xr.Dataset or xr.DataArray.
    """
    return horizontal_block_reduce(
        obj,
        coarsening_factor,
        np.median,
        x_dim=x_dim,
        y_dim=y_dim
    )


def block_coarsen(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1',
    method: str = 'sum'
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by performing an operation over blocks.

    Args:
        obj: Input DataArray or Dataset.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        method: Name of coarsening method to use.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    coarsen_object = obj.coarsen(coarsen_kwargs)
    result = getattr(coarsen_object, method)()

    return result


def block_edge_sum(
    obj: Union[xr.Dataset, xr.DataArray],
    coarsening_factor: int,
    x_dim: Hashable = 'xaxis_1',
    y_dim: Hashable = 'yaxis_1',
    edge: str = 'x'
) -> Union[xr.Dataset, xr.DataArray]:
    """Coarsen a DataArray or Dataset by summing along a block edge.

    Mainly meant for coarse-graining the dx and dy variables from the original
    grid_spec.

    Args:
        obj: Input DataArray or Dataset.
        coarsening_factor: Integer coarsening factor to use.
        x_dim: x dimension name (default 'xaxis_1').
        y_dim: y dimension name (default 'yaxis_1').
        edge: Grid cell side to coarse-grain along {'x', 'y'}.

    Returns:
        xr.Dataset or xr.DataArray.
    """
    if edge == 'x':
        coarsen_dim = x_dim
        downsample_dim = y_dim
    elif edge == 'y':
        coarsen_dim = y_dim
        downsample_dim = x_dim
    else:
        raise ValueError(f"'edge' most be either 'x' or 'y'; got {edge}.")

    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = obj.coarsen(coarsen_kwargs).sum()
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return result
