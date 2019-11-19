"""Tools for working with cubedsphere data"""
import logging
from os.path import join

import dask
import numpy as np
import xarray as xr

from skimage.measure import block_reduce

NUM_TILES = 6
SUBTILE_FILE_PATTERN = '{prefix}.tile{tile:d}.nc.{subtile:04d}'


def shift_edge_var_to_center(edge_var: xr.DataArray):
    # assumes C or D grid
    if 'grid_y' in edge_var.dims:
        return _rename_xy_coords(0.5 * (edge_var + edge_var.shift(grid_y=1))[:,  1:, :])
    elif 'grid_x' in edge_var.dims:
        return _rename_xy_coords(0.5 * (edge_var + edge_var.shift(grid_x=1))[:, :, 1:])
    else:
        raise ValueError(
            'Variable to shift to center must be centered on one horizontal axis and edge-valued on the other.')


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


def coarsen_coords(factor, tile, dims):
    return {
        key: ((tile[key][::factor] - 1) // factor + 1).astype(int).astype(np.float32)
        for key in dims
    }


def add_coordinates(reference_ds, coarsened_ds, factor, x_dim, y_dim):
    new_coordinates = coarsen_coords(factor, reference_ds, [x_dim, y_dim])

    if isinstance(coarsened_ds, xr.DataArray):
        result = coarsened_ds.assign_coords(
            **new_coordinates
        ).rename(coarsened_ds.name)
    else:
        result = coarsened_ds.assign_coords(**new_coordinates)
    return result


def weighted_block_average(
    da,
    weights,
    coarsening_factor,
    x_dim='xaxis_1',
    y_dim='yaxis_2'
):
    """Coarsen a DataArray or Dataset through weighted block averaging.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    weights : xr.DataArray
        Weights (e.g. area or pressure thickness)
    coarsening_factor : int
        Coarsening factor to use
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name

    Returns
    -------
    xr.DataArray

    Warnings
    --------
    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    result = ((da * weights).coarsen(**coarsen_kwargs).sum() /
              weights.coarsen(**coarsen_kwargs).sum())

    return add_coordinates(da, result, coarsening_factor, x_dim, y_dim)


def edge_weighted_block_average(
    da,
    spacing,
    coarsening_factor,
    x_dim='xaxis_1',
    y_dim='yaxis_1',
    edge='x'
):
    """Coarsen a DataArray or Dataset along a block edge.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    spacing : xr.DataArray
        Spacing of grid cells in the coarsening dimension
    coarsening_factor : int
        Coarsening factor to use
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name
    edge : str {'x', 'y'}
        Dimension to coarse-grain along

    Returns
    -------
    xr.DataArray

    Warnings
    --------
    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.
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
        (spacing * da).coarsen(coarsen_kwargs).sum() /
        spacing.coarsen(coarsen_kwargs).sum()
    )
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return add_coordinates(
        da, result, coarsening_factor, coarsen_dim, downsample_dim
    )


def _general_block_median(da, block_sizes):
    """A dask compatible block median function that works on
    DataArrays.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray
    block_sizes : dict
        Dictionary of dimension names mapping to block sizes

    Returns
    -------
    xr.DataArray
    """
    def func(data, ordered_block_sizes, chunks):
        if isinstance(data, dask.array.Array):
            return dask.array.map_blocks(
                block_reduce,
                data,
                ordered_block_sizes,
                np.median,
                dtype=np.float32,
                chunks=chunks
            )
        else:
            return block_reduce(data, ordered_block_sizes, np.median)

    # TODO: add some error checking to assert that block_sizes[dim] evenly
    # divides each chunk in the original array.
    chunks = []
    for dim in da.dims:
        new_chunks = tuple(
            np.array(da.chunks[da.get_axis_num(dim)]) // block_sizes[dim]
        )
        chunks.append(new_chunks)
    chunks = tuple(chunks)

    ordered_block_sizes = tuple(block_sizes[dim] for dim in da.dims)

    result = xr.apply_ufunc(
        func,
        da,
        ordered_block_sizes,
        chunks,
        input_core_dims=[da.dims, [], []],
        output_core_dims=[da.dims],
        dask='allowed',
        exclude_dims=set(da.dims)
    )

    # Restore coordinates for dimensions that did not change size
    for dim, size in block_sizes.items():
        if size == 1:
            result[dim] = da[dim]

    return result


def block_median(ds, coarsening_factor, x_dim='xaxis_1', y_dim='yaxis_1'):
    """Coarsen a DataArray or Dataset by taking the median over blocks.

    Mainly meant for coarse-graining sfc_data variables.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    coarsening_factor : int
        Coarsening factor to use
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name

    Returns
    -------
    xr.DataArray
    """
    def _block_median_da(da, coarsening_factor, x_dim, y_dim):
        block_sizes = {}
        for dim in da.dims:
            if dim in [x_dim, y_dim]:
                block_sizes[dim] = coarsening_factor
            else:
                block_sizes[dim] = 1

        return _general_block_median(da, block_sizes)

    if isinstance(ds, xr.Dataset):
        results = []
        for da in ds.values():
            if x_dim in da.dims and y_dim in da.dims:
                results.append(_block_median_da(
                    da,
                    coarsening_factor,
                    x_dim,
                    y_dim)
                )
            else:
                results.append(da)

        result = xr.merge(results)
    else:
        result = _block_median_da(da, coarsening_factor, x_dim, y_dim)

    return add_coordinates(da, result, coarsening_factor, x_dim, y_dim)


def block_coarsen(
    da,
    coarsening_factor,
    x_dim='xaxis_1',
    y_dim='yaxis_1',
    method='sum'
):
    """Coarsen a DataArray or Dataset by performing an operation over blocks.

    Mainly meant for coarse-graining the area variable from the original
    grid_spec.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    coarsening_factor : int
        Coarsening factor to use
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name
    method : str
        Coarsening method

    Returns
    -------
    xr.DataArray

    Warnings
    --------
    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.
    """
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    coarsen_object = da.coarsen(**coarsen_kwargs)
    result = getattr(coarsen_object, method)()

    return add_coordinates(da, result, coarsening_factor, x_dim, y_dim)


def block_edge_sum(
    da,
    coarsening_factor,
    x_dim='xaxis_1',
    y_dim='yaxis_1',
    edge='x'
):
    """Coarsen a DataArray or Dataset by summing along a block edge.

    Mainly meant for coarse-graining the dx and dy variables from the original
    grid_spec.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    coarsening_factor : int
        Coarsening factor to use
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name
    edge : str {'x', 'y'}
        Dimension to coarse-grain along

    Returns
    -------
    xr.DataArray

    Warnings
    --------
    Note that this function assumes that the x and y dimension names of the
    input DataArray and weights are the same.
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
    coarsened = da.coarsen(coarsen_kwargs).sum()
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return add_coordinates(
        da, result, coarsening_factor, coarsen_dim, downsample_dim
    )


def save_tiles_separately(sfc_data, prefix, output_directory):

    #TODO: move to vcm.cubedsphere
    for i in range(6):
        output_path = join(output_directory, f"{prefix}.tile{i+1}.nc")
        logging.info(f"saving data to {output_path}")
        sfc_data.isel(tile=i).to_netcdf(output_path)