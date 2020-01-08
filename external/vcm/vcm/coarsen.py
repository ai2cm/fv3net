"""
Utilities for coarse-graining restart data and directories
"""
import logging
import os
from os.path import join

import dask.bag as db
import numpy as np
import pandas as pd
import xarray as xr
from toolz import curry

from .complex_sfc_data_coarsening import coarse_grain_sfc_data_complex
from .cubedsphere import (
    block_coarsen,
    block_edge_sum,
    block_median,
    coarsen_coords,
    edge_weighted_block_average,
    open_cubed_sphere,
    regrid_to_area_weighted_pressure,
    regrid_to_edge_weighted_pressure,
    weighted_block_average,
)
from .calc.thermo import hydrostatic_dz, height_at_interface, dz_and_top_to_phis
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    FV_CORE_X_CENTER,
    FV_CORE_Y_CENTER,
    FV_CORE_X_OUTER,
    FV_CORE_Y_OUTER,
    FV_TRACER_X_CENTER,
    FV_TRACER_Y_CENTER,
    RESTART_Z_CENTER,
)

TILES = range(1, 7)
OUTPUT_CATEGORY_NAMES = {
    "fv_core.res": "fv_core_coarse.res",
    "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
    "fv_tracer.res": "fv_tracer_coarse.res",
    "sfc_data": "sfc_data_coarse",
}
SOURCE_DATA_PATTERN = "{timestep}/{timestep}.{category}"
DATA_PATTERN = "{prefix}.{category}.tile{tile}.nc"
CATEGORY_LIST = ["fv_core.res", "fv_srf_wnd.res", "fv_tracer.res", "sfc_data"]


def integerize(x):
    return np.round(x).astype(x.dtype)


# TODO: rename this to coarsen_by_area. It is not specific to surface data
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
    coarse_coords = coarsen_coords(factor, data, ["xaxis_1", "yaxis_1"])

    # special hack for SLMASK (should be integer quantity)
    coarsened["slmsk"] = integerize(coarsened.slmsk)

    return coarsened.assign_coords(**coarse_coords).assign_attrs(
        {"coarsening_factor": factor, "coarsening_method": method}
    )


# TODO: fix this name. it loads the data with area
def load_tile_proc(tile, subtile, path, grid_path):
    grid_spec_to_data_renaming = {COORD_X_CENTER: "xaxis_1", COORD_Y_CENTER: "yaxis_1"}
    grid = xr.open_dataset(grid_path)

    area = grid.area.rename(grid_spec_to_data_renaming)
    pane = xr.open_dataset(path)

    data = xr.merge([pane, area])

    return data


# TODO use vcm.cubedsphere for this
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

    # res = bag.map(process)
    # res = res.compute(scheduler='single-threaded')

    procs = bag.map(process).groupby(tile).map(combine_subtiles).map(lambda x: x[1])

    return xr.concat(procs.compute(), dim="tile")


def coarse_grain_fv_core(ds, delp, area, dx, dy, coarsening_factor):
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
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["phis", "delp", "DZ"]
    mass_weighted_vars = ["W", "T"]
    dx_edge_weighted_vars = ["u"]
    dy_edge_weighted_vars = ["v"]

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    edge_weighted_x = edge_weighted_block_average(
        ds[dx_edge_weighted_vars],
        dx,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
    )

    edge_weighted_y = edge_weighted_block_average(
        ds[dy_edge_weighted_vars],
        dy,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
    )

    return xr.merge([area_weighted, mass_weighted, edge_weighted_x, edge_weighted_y])


def coarse_grain_fv_core_on_pressure(ds, delp, area, dx, dy, coarsening_factor):
    """Coarse grain a set of fv_core restart files, averaging on surfaces of
    constant pressure (except for delp, DZ and phis which are averaged on model
    surfaces).

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
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["phis", "delp", "DZ"]
    mass_weighted_vars = ["W", "T"]
    dx_edge_weighted_vars = ["u"]
    dy_edge_weighted_vars = ["v"]

    area_pressure_regridded, masked_area = regrid_to_area_weighted_pressure(
        ds[mass_weighted_vars],
        delp,
        area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    dx_pressure_regridded, masked_dx = regrid_to_edge_weighted_pressure(
        ds[dx_edge_weighted_vars],
        delp,
        dx,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
    )

    dy_pressure_regridded, masked_dy = regrid_to_edge_weighted_pressure(
        ds[dy_edge_weighted_vars],
        delp,
        dy,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
    )

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    masked_mass = delp * masked_area
    mass_weighted = weighted_block_average(
        area_pressure_regridded,
        masked_mass,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_CENTER,
    )

    edge_weighted_x = edge_weighted_block_average(
        dx_pressure_regridded,
        masked_dx,
        coarsening_factor,
        x_dim=FV_CORE_X_CENTER,
        y_dim=FV_CORE_Y_OUTER,
        edge="x",
    )

    edge_weighted_y = edge_weighted_block_average(
        dy_pressure_regridded,
        masked_dy,
        coarsening_factor,
        x_dim=FV_CORE_X_OUTER,
        y_dim=FV_CORE_Y_CENTER,
        edge="y",
    )

    return xr.merge([area_weighted, mass_weighted, edge_weighted_x, edge_weighted_y])


def coarse_grain_fv_tracer(ds, delp, area, coarsening_factor):
    """Coarse grain a set of fv_tracer restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["cld_amt"]
    mass_weighted_vars = [
        "sphum",
        "liq_wat",
        "rainwat",
        "ice_wat",
        "snowwat",
        "graupel",
        "o3mr",
        "sgs_tke",
    ]

    area_weighted = weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    return xr.merge([area_weighted, mass_weighted])


def coarse_grain_fv_tracer_on_pressure(ds, delp, area, coarsening_factor):
    """Coarse grain a set of fv_tracer restart files, averaging on surfaces of
    constant pressure.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_tracer restart files
    delp : xr.DataArray
        Pressure thicknesses
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["cld_amt"]
    mass_weighted_vars = [
        "sphum",
        "liq_wat",
        "rainwat",
        "ice_wat",
        "snowwat",
        "graupel",
        "o3mr",
        "sgs_tke",
    ]

    ds_regridded, masked_area = regrid_to_area_weighted_pressure(
        ds,
        delp,
        area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    area_weighted = weighted_block_average(
        ds_regridded[area_weighted_vars],
        masked_area,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    masked_mass = delp * masked_area
    mass_weighted = weighted_block_average(
        ds_regridded[mass_weighted_vars],
        masked_mass,
        coarsening_factor,
        x_dim=FV_TRACER_X_CENTER,
        y_dim=FV_TRACER_Y_CENTER,
    )

    return xr.merge([area_weighted, mass_weighted])


def coarse_grain_fv_srf_wnd(ds, area, coarsening_factor):
    """Coarse grain a set of fv_srf_wnd restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of fv_srf_wnd restart files
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    area_weighted_vars = ["u_srf", "v_srf"]
    return weighted_block_average(
        ds[area_weighted_vars],
        area,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )


def impose_hydrostatic_balance(ds_fv_core, ds_fv_tracer, dim=RESTART_Z_CENTER):
    """Compute layer thicknesses assuming hydrostatic balance and adjust
    surface geopotential in order to maintain same model top height.

    Args:
        ds_fv_core (xr.Dataset): fv_core restart category Dataset
        ds_fv_tracer (xr.Dataset): fv_tracer restart category Dataset
        dim (str): vertical dimension name (default "zaxis_1")

    Returns:
        xr.Dataset: ds_fv_core with hydrostatic DZ and adjusted phis
    """
    height = height_at_interface(
        ds_fv_core["DZ"], ds_fv_core["phis"], dim_center=dim, dim_outer=dim
    )
    height_top = height.isel({dim: 0})
    dz = hydrostatic_dz(
        ds_fv_core["T"],
        ds_fv_tracer["sphum"].rename({FV_TRACER_Y_CENTER: FV_CORE_Y_CENTER}),
        ds_fv_core["delp"],
        dim=dim,
    )
    return ds_fv_core.assign(DZ=dz, phis=dz_and_top_to_phis(height_top, dz, dim=dim))


def coarse_grain_sfc_data(ds, area, coarsening_factor, version="simple"):
    """Coarse grain a set of sfc_data restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of sfc_data restart files
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use
    version : str
        Version of the method to use {'simple', 'complex'}

    Returns
    -------
    xr.Dataset
    """
    if version == "simple":
        result = block_median(ds, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_1")
        result["slmsk"] = integerize(result.slmsk)
        return result
    elif version == "complex":
        return coarse_grain_sfc_data_complex(ds, area, coarsening_factor)
    else:
        raise ValueError(
            f"Currently the only supported versions are 'simple' and 'complex'. "
            "Got {version}."
        )


def coarse_grain_grid_spec(
    ds,
    coarsening_factor,
    x_dim_unstaggered=COORD_X_CENTER,
    y_dim_unstaggered=COORD_Y_CENTER,
    x_dim_staggered=COORD_X_OUTER,
    y_dim_staggered=COORD_Y_OUTER,
):
    coarse_dx = block_edge_sum(
        ds.dx, coarsening_factor, x_dim_unstaggered, y_dim_staggered, "x"
    )
    coarse_dy = block_edge_sum(
        ds.dy, coarsening_factor, x_dim_staggered, y_dim_unstaggered, "y"
    )
    coarse_area = block_coarsen(
        ds.area, coarsening_factor, x_dim_unstaggered, y_dim_unstaggered
    )

    return xr.merge(
        [
            coarse_dx.rename(ds.dx.name),
            coarse_dy.rename(ds.dy.name),
            coarse_area.rename(ds.area.name),
        ]
    )


def sync_dimension_order(a, b):
    for var in a:
        a[var] = a[var].transpose(*b[var].dims)
    return a


def coarsen_grid_spec(
    input_grid_spec,
    coarsening_factor,
    output_filename,
    x_dim_unstaggered=COORD_X_CENTER,
    y_dim_unstaggered=COORD_Y_CENTER,
    x_dim_staggered=COORD_X_OUTER,
    y_dim_staggered=COORD_Y_OUTER,
):
    tile = pd.Index(TILES, name="tile")
    native_grid_spec = xr.open_mfdataset(input_grid_spec, concat_dim=tile)
    result = coarse_grain_grid_spec(
        native_grid_spec,
        coarsening_factor,
        x_dim_unstaggered,
        y_dim_unstaggered,
        x_dim_staggered,
        y_dim_staggered,
    )
    result.to_netcdf(output_filename)


def coarsen_restart_file_category(
    timestep,
    native_category_name,
    coarsening_factor,
    coarse_grid_spec,
    native_grid_spec,
    source_data_prefix,
    output_files,
    source_data_pattern=SOURCE_DATA_PATTERN,
):
    category = OUTPUT_CATEGORY_NAMES[native_category_name]
    grid_spec = xr.open_dataset(coarse_grid_spec, chunks={"tile": 1})
    tile = pd.Index(TILES, name="tile")
    source = open_cubed_sphere(
        os.path.join(
            source_data_prefix,
            source_data_pattern.format(timestep=timestep, category=category),
        )
    )

    if category == "fv_core_coarse.res":
        coarsened = coarse_grain_fv_core(
            source,
            source.delp,
            grid_spec.area.rename(
                {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
            ),
            grid_spec.dx.rename(
                {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_OUTER: FV_CORE_Y_OUTER}
            ),
            grid_spec.dy.rename(
                {COORD_X_OUTER: FV_CORE_X_OUTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
            ),
            coarsening_factor,
        )
    elif category == "fv_srf_wnd_coarse.res":
        coarsened = coarse_grain_fv_srf_wnd(
            source,
            grid_spec.area.rename(
                {COORD_X_CENTER: "xaxis_1", COORD_Y_CENTER: "yaxis_1"}
            ),
            coarsening_factor,
        )
    elif category == "fv_tracer_coarse.res":
        fv_core = open_cubed_sphere(
            os.path.join(
                source_data_prefix,
                source_data_pattern.format(
                    timestep=timestep, category="fv_core_coarse.res"
                ),
            )
        )
        coarsened = coarse_grain_fv_tracer(
            source,
            fv_core.delp.rename({FV_CORE_Y_CENTER: FV_TRACER_Y_CENTER}),
            grid_spec.area.rename(
                {COORD_X_CENTER: FV_TRACER_X_CENTER, COORD_Y_CENTER: FV_TRACER_Y_CENTER}
            ),
            coarsening_factor,
        )
    elif category == "sfc_data":
        native_grid_spec = xr.open_mfdataset(native_grid_spec, concat_dim=tile)
        coarsened = coarse_grain_sfc_data(
            source,
            native_grid_spec.area.rename(
                {COORD_X_CENTER: "xaxis_1", COORD_Y_CENTER: "yaxis_1"}
            ),
            coarsening_factor,
        )
    else:
        raise ValueError(
            f"Cannot coarse grain files for unknown 'category'," "{category}."
        )

    coarsened = sync_dimension_order(coarsened, source)
    for tile, file in zip(TILES, output_files):
        coarsened.sel(tile=tile).drop("tile").to_netcdf(file)


def coarsen_restarts_on_pressure(
    coarsening_factor,
    grid_spec_prefix,
    source_data_prefix,
    output_data_prefix,
    data_pattern=DATA_PATTERN,
):
    """ Coarsen a complete set of restart files, averaging on pressure levels and
    using the 'complex' surface coarsening method

    Args:
        coarsening_factor (int): Amount to coarsen, e.g. 8 for C384 to C48.
        grid_spec_prefix (str): Path to grid_spec at same resolution as source data.
            Must include everything up to '.tile*.nc'
        source_data_prefix (str): Prefix of path to fine-resolution restart files.
        output_data_prefix (str): Prefix of path where coarsened restart files will be
            saved.
        data_pattern (str, optional): pattern for source and output data file naming
            convention. Defaults to "{prefix}.{category}.tile{tile}.nc". Must include
            prefix, category and tile.
    """
    tiles = pd.Index(range(6), name="tile")
    filename = grid_spec_prefix + ".tile*.nc"
    grid_spec = xr.open_mfdataset(filename, concat_dim=[tiles], combine="nested")

    source = {}
    coarsened = {}
    for category in CATEGORY_LIST:
        filename = data_pattern.format(
            prefix=source_data_prefix,
            category=OUTPUT_CATEGORY_NAMES[category],
            tile="*",
        )
        source[category] = xr.open_mfdataset(
            filename, concat_dim=[tiles], combine="nested"
        )

        if category == "fv_core.res":
            coarsened[category] = coarse_grain_fv_core_on_pressure(
                source[category],
                source[category].delp,
                grid_spec.area.rename(
                    {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
                ),
                grid_spec.dx.rename(
                    {COORD_X_CENTER: FV_CORE_X_CENTER, COORD_Y_OUTER: FV_CORE_Y_OUTER}
                ),
                grid_spec.dy.rename(
                    {COORD_X_OUTER: FV_CORE_X_OUTER, COORD_Y_CENTER: FV_CORE_Y_CENTER}
                ),
                coarsening_factor,
            )
        elif category == "fv_srf_wnd.res":
            coarsened[category] = coarse_grain_fv_srf_wnd(
                source[category],
                grid_spec.area.rename(
                    {COORD_X_CENTER: "xaxis_1", COORD_Y_CENTER: "yaxis_1"}
                ),
                coarsening_factor,
            )
        elif category == "fv_tracer.res":
            coarsened[category] = coarse_grain_fv_tracer_on_pressure(
                source[category],
                source["fv_core.res"].delp.rename(
                    {FV_CORE_Y_CENTER: FV_TRACER_Y_CENTER}
                ),
                grid_spec.area.rename(
                    {
                        COORD_X_CENTER: FV_TRACER_X_CENTER,
                        COORD_Y_CENTER: FV_TRACER_Y_CENTER,
                    }
                ),
                coarsening_factor,
            )
        elif category == "sfc_data":
            coarsened[category] = coarse_grain_sfc_data_complex(
                source[category],
                grid_spec.area.rename(
                    {COORD_X_CENTER: "xaxis_1", COORD_Y_CENTER: "yaxis_1"}
                ),
                coarsening_factor,
            )

    coarsened["fv_core.res"] = impose_hydrostatic_balance(
        coarsened["fv_core.res"], coarsened["fv_tracer.res"]
    )

    for category in CATEGORY_LIST:
        coarsened[category] = sync_dimension_order(
            coarsened[category], source[category]
        )
        _save_restart_category(
            coarsened[category], output_data_prefix, category, data_pattern
        )


def _save_restart_category(ds, prefix, category, data_pattern):
    for tile in TILES:
        filename = data_pattern.format(prefix=prefix, category=category, tile=tile)
        ds.isel(tile=tile - 1).drop("tile", errors="ignore").to_netcdf(filename)

