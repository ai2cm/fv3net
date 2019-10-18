import xarray as xr
import os
from itertools import product
import dask
import dask.bag as db
from os.path import join
from skimage.measure import block_reduce
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
        {'coarsening_factor': 32, 'coarsening_method': method}
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
    return tile, _combine_subtiles(subtiles).assign(tile=tile)


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

    procs = (
        bag
        .map(process)
        .groupby(tile)
        .map(combine_subtiles)
        .map(lambda x: x[1])
    )

    return xr.concat(procs.compute(), dim='tile')


def add_coordinates(ds, x_dim, y_dim):
    new_coordinates = {
        x_dim: np.arange(1., ds.sizes[x_dim] + 1.).astype(np.float32),
        y_dim: np.arange(1., ds.sizes[y_dim] + 1.).astype(np.float32)
    }

    if isinstance(ds, xr.DataArray):
        result = ds.assign_coords(**new_coordinates).rename(ds.name)
    else:
        result = ds.assign_coords(**new_coordinates)
    return result


def weighted_block_average(
    da,
    weights,
    target_resolution,
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
    target_resolution : int
        Target resolution to coarsen data to
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
    if da.sizes[x_dim] % target_resolution != 0:
        raise ValueError(f"'target_resolution' {target_resolution} specified "
                         "does not evenly divide the data resolution")

    coarsening_factor = da.sizes[x_dim] // target_resolution
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    result = ((da * weights).coarsen(**coarsen_kwargs).sum() /
              weights.coarsen(**coarsen_kwargs).sum())

    return add_coordinates(result, x_dim, y_dim)


def edge_weighted_block_average(
    da,
    spacing,
    target_resolution,
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
    target_resolution : int
        Target resolution to coarsen data to
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

    if da.sizes[coarsen_dim] % target_resolution != 0:
        raise ValueError(
            f"'target_resolution' {target_resolution} specified "
            "does not evenly divide the data resolution."
        )

    coarsening_factor = da.sizes[coarsen_dim] // target_resolution
    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = (
        (spacing * da).coarsen(coarsen_kwargs).sum() /
        spacing.coarsen(coarsen_kwargs).sum()
    )
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return add_coordinates(result, coarsen_dim, downsample_dim)


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


def block_median(ds, target_resolution, x_dim='xaxis_1', y_dim='yaxis_1'):
    """Coarsen a DataArray or Dataset by taking the median over blocks.

    Mainly meant for coarse-graining sfc_data variables.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    target_resolution : int
        Target resolution to coarsen data to
    x_dim : str
        x dimension name
    y_dim : str
        y dimension name

    Returns
    -------
    xr.DataArray
    """
    def _block_median_da(da, target_resolution, x_dim, y_dim):
        factor = da.sizes[x_dim] // target_resolution
        block_sizes = {}
        for dim in da.dims:
            if dim in [x_dim, y_dim]:
                block_sizes[dim] = factor
            else:
                block_sizes[dim] = 1

        return _general_block_median(da, block_sizes)

    if ds.sizes[x_dim] % target_resolution != 0:
        raise ValueError(f"'target_resolution' {target_resolution} specified "
                         "does not evenly divide the data resolution")

    if isinstance(ds, xr.Dataset):
        results = []
        for da in ds.values():
            if x_dim in da.dims and y_dim in da.dims:
                results.append(_block_median_da(
                    da,
                    target_resolution,
                    x_dim,
                    y_dim)
                )
            else:
                results.append(da)

        result = xr.merge(results)
    else:
        result = _block_median_da(da, target_resolution, x_dim, y_dim)

    return add_coordinates(result, x_dim, y_dim)


def block_sum(
    da,
    target_resolution,
    x_dim='xaxis_1',
    y_dim='yaxis_1',
):
    """Coarsen a DataArray or Dataset by summing over blocks.

    Mainly meant for coarse-graining the area variable from the original
    grid_spec.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input DataArray or Dataset
    target_resolution : int
        Target resolution to coarsen data to
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
    if da.sizes[x_dim] % target_resolution != 0:
        raise ValueError(f"'target_resolution' {target_resolution} specified "
                         "does not evenly divide the data resolution")

    coarsening_factor = da.sizes[x_dim] // target_resolution
    coarsen_kwargs = {x_dim: coarsening_factor, y_dim: coarsening_factor}
    result = da.coarsen(**coarsen_kwargs).sum()

    return add_coordinates(result, x_dim, y_dim)


def block_edge_sum(
    da,
    target_resolution,
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
    target_resolution : int
        Target resolution to coarsen data to
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

    if da.sizes[coarsen_dim] % target_resolution != 0:
        raise ValueError(
            f"'target_resolution' {target_resolution} specified "
            "does not evenly divide the data resolution."
        )

    coarsening_factor = da.sizes[coarsen_dim] // target_resolution
    coarsen_kwargs = {coarsen_dim: coarsening_factor}
    coarsened = da.coarsen(coarsen_kwargs).sum()
    downsample_kwargs = {downsample_dim: slice(None, None, coarsening_factor)}
    result = coarsened.isel(downsample_kwargs)

    return add_coordinates(result, coarsen_dim, downsample_dim)



def coarse_grain_fv_core(ds, delp, area, dx, dy, target_resolution):
    """Coarse grain a set of fv_core restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_core restart files
    area : xr.DataArray
        Area weights
    dx : xr.DataArray
        x edge lengths
    dy : xr.DataArray
        y edge lengths
    target_resolution : int
        Target resolution to coarsen to

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ['phis', 'delp', 'DZ']
    mass_weighted_vars = ['W', 'T']
    dx_edge_weighted_vars = ['u']
    dy_edge_weighted_vars = ['v']

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_2'
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_2'
    )

    edge_weighted_x = edge_weighted_block_average(
        ds[dx_edge_weighted_vars],
        dx,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_1',
        edge='x'
    )

    edge_weighted_y = edge_weighted_block_average(
        ds[dy_edge_weighted_vars],
        dy,
        target_resolution,
        x_dim='xaxis_2',
        y_dim='yaxis_2',
        edge='y'
    )

    return xr.merge(
        [area_weighted, mass_weighted, edge_weighted_x, edge_weighted_y]
    )


def coarse_grain_fv_tracer(ds, delp, area, target_resolution):
    """Coarse grain a set of fv_tracer restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    target_resolution : int
        Target resolution to coarsen to

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ['cld_amt']
    mass_weighted_vars = [
        'sphum',
        'liq_wat',
        'rainwat',
        'ice_wat',
        'snowwat',
        'graupel',
        'o3mr',
        'sgs_tke'
    ]

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_1'
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_1'
    )

    return xr.merge([area_weighted, mass_weighted])


def coarse_grain_fv_srf_wnd(ds, area, target_resolution):
    """Coarse grain a set of fv_srf_wnd restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_srf_wnd restart files
    area : xr.DataArray
        Area weights
    target_resolution : int
        Target resolution to coarsen to

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ['u_srf', 'v_srf']
    return weighted_block_average(
        ds[area_weighted_vars],
        area,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_1'
    )


def coarse_grain_sfc_data(ds, area, target_resolution):
    """Coarse grain a set of sfc_data restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of sfc_data restart files
    area : xr.DataArray
        Area weights
    target_resolution : int
        Target resolution to coarsen to

    Returns
    -------
    xr.Dataset
    """
    result = block_median(
        ds,
        target_resolution,
        x_dim='xaxis_1',
        y_dim='yaxis_1'
    )

    result['slmsk'] = np.round(result.slmsk).astype(result.slmsk.dtype)
    return result
