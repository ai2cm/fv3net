"""
Utilities for coarse-graining restart data and directories
"""
import logging
import os
from os.path import join

import dask.array as dask_array
import dask.bag as db
import numpy as np
import pandas as pd
import xarray as xr
from toolz import curry

from .casting import doubles_to_floats
from .cubedsphere import (
    block_coarsen,
    block_edge_sum,
    block_median,
    block_mode,
    block_upsample,
    coarsen_coords,
    edge_weighted_block_average,
    open_cubed_sphere,
    weighted_block_average,
)

TILES = range(1, 7)
OUTPUT_CATEGORY_NAMES = {
    "fv_core.res": "fv_core_coarse.res",
    "fv_srf_wnd.res": "fv_srf_wnd_coarse.res",
    "fv_tracer.res": "fv_tracer_coarse.res",
    "sfc_data": "sfc_data",
}
SOURCE_DATA_PATTERN = "{timestep}/{timestep}.{category}"

# Define global constants to specify how each of the sfc_data variables should
# be coarsened.
AREA_MEAN = [
    "tsea",
    "alvsf",
    "alvwf",
    "alnsf",
    "alnwf",
    "facsf",
    "facwf",
    "f10m",
    "t2m",
    "q2m",
    "uustar",
    "ffmm",
    "ffhh",
    "tprcp",
    "snwdph",
]
AREA_MEAN_OVER_DOMINANT_SFC_TYPE = [
    "tg3",
    "vfrac",
    "fice",
    "sncovr",
]
AREA_AND_VFRAC_MEAN_OVER_DOMINANT_SFC_AND_VTYPE = ["canopy", "zorl"]
AREA_MEAN_OVER_DOMINANT_SFC_AND_STYPE = ["smc", "slc", "stc"]
MODE = ["slmsk", "srflag"]
MODE_OVER_DOMINANT_SFC_TYPE = ["slope", "vtype", "stype"]
AREA_AND_SNCOVR_MEAN = ["sheleg"]
AREA_AND_FICE_MEAN = ["hice"]
MINIMUM_OVER_DOMINANT_SFC_TYPE = ["shdmin"]
MAXIMUM_OVER_DOMINANT_SFC_TYPE = ["shdmax", "snoalb"]


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
    grid_spec_to_data_renaming = {"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}
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
        x_dim="xaxis_1",
        y_dim="yaxis_2",
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_2",
    )

    edge_weighted_x = edge_weighted_block_average(
        ds[dx_edge_weighted_vars],
        dx,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
        edge="x",
    )

    edge_weighted_y = edge_weighted_block_average(
        ds[dy_edge_weighted_vars],
        dy,
        coarsening_factor,
        x_dim="xaxis_2",
        y_dim="yaxis_2",
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
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )

    mass = delp * area
    mass_weighted = weighted_block_average(
        ds[mass_weighted_vars],
        mass,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
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


def coarse_grain_sfc_data(ds, area, coarsening_factor):
    """Coarse grain a set of sfc_data restart files.

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset; assumed to be from a set of sfc_data restart files
    area : xr.DataArray
        Area weights
    coarsening_factor : int
        Coarsening factor to use

    Returns
    -------
    xr.Dataset
    """
    result = block_median(ds, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_1")

    result["slmsk"] = integerize(result.slmsk)
    return result


def _apply_surface_chgres_corrections(ds: xr.Dataset) -> xr.Dataset:
    """This function applies the corrections noted in surface_chgres.F90
    to the variables in the coarsened surface Dataset.
    
    See https://github.com/VulcanClimateModeling/fv3gfs/blob/
    master/sorc/global_chgres.fd/surface_chgres.f90 for more information.
    """
    # 1. Clip tsea and tg3 at 273.16 K if a cell contains land ice.
    vtype_land_ice = 15.0
    freezing_temperature = 273.16

    clipped_tsea = ds.tsea.where(
        ds.tsea < freezing_temperature, other=freezing_temperature
    )
    clipped_t3g = ds.tg3.where(
        ds.tg3 < freezing_temperature, other=freezing_temperature
    )

    tsea = xr.where(ds.vtype == vtype_land_ice, clipped_tsea, ds.tsea)
    tg3 = xr.where(ds.vtype == vtype_land_ice, clipped_t3g, ds.tg3)

    ds["tsea"] = tsea
    ds["tg3"] = tg3

    # 2. If a cell contains land ice, make sure the soil type is ice
    stype_land_ice = 16.0
    stype = xr.where(ds.vtype == vtype_land_ice, stype_land_ice, ds.stype)
    ds["stype"] = stype

    # 3. If a cell does not contain vegetation, i.e. if shdmin < 0.011,
    # then set the canopy moisture content to zero.
    threshold = 0.011
    canopy = xr.where(ds.shdmin < threshold, 0.0, ds.canopy)
    ds["canopy"] = canopy

    # 4. If a cell contains land ice, then shdmin is set to zero.
    shdmin = xr.where(ds.vtype == vtype_land_ice, 0.0, ds.shdmin)
    ds["shdmin"] = shdmin

    return ds


def _block_upsample_and_rechunk(
    da: xr.DataArray, reference_da: xr.DataArray, upsampling_factor: int
) -> xr.DataArray:
    """Upsample DataArray and rechunk to match reference DataArray's chunks.

    This is needed, because block_upsample leads to unnecessarily small chunk
    sizes when applied to dask arrays, which degrades performance if not
    corrected.

    This private function is strictly for use in
    coarse_grain_sfc_data_complicated; it implicitly assumes that the
    reference_da has dimensions that match the upsampled da.
    """
    result = block_upsample(da, upsampling_factor)
    if isinstance(da.data, dask_array.Array):
        return result.chunk(reference_da.chunks)
    else:
        return result


def coarse_grain_sfc_data_complicated(
    ds: xr.Dataset, area: xr.DataArray, coarsening_factor: int
) -> xr.Dataset:
    """Coarse grain a set of sfc_data restart files using the 'complicated'
    method.

    See:

    https://paper.dropbox.com/doc/Downsampling-restart-fields-for-the-
    Noah-land-surface-model--Ap8z8JnU1OH4HwNZ~PY4z~p~Ag-OgkIEYSn0g4X9CCZxn5dy
    
    for a description of what this procedure does.

    Args:
        ds: Input Dataset.
        area: DataArray with the surface area of each grid cell; it is assumed
            its horizontal dimension names are 'xaxis_1' and 'yaxis_1'.
        coarsening_factor: Integer coarsening factor to use.

    Returns:
        xr.Dataset
    """
    mode = block_mode(ds[MODE], coarsening_factor)

    upsampled_slmsk = _block_upsample_and_rechunk(
        mode.slmsk, ds.slmsk, coarsening_factor
    )
    is_dominant_surface_type = ds.slmsk == upsampled_slmsk

    mode_over_dominant_surface_type = block_mode(
        ds[MODE_OVER_DOMINANT_SFC_TYPE].where(is_dominant_surface_type),
        coarsening_factor,
    )

    upsampled_vtype = _block_upsample_and_rechunk(
        mode_over_dominant_surface_type.vtype, ds.vtype, coarsening_factor
    )
    is_dominant_vtype = ds.vtype == upsampled_vtype
    is_dominant_vegetation_and_surface_type = (
        is_dominant_surface_type & is_dominant_vtype
    )

    upsampled_stype = _block_upsample_and_rechunk(
        mode_over_dominant_surface_type.stype, ds.stype, coarsening_factor
    )
    is_dominant_stype = ds.stype == upsampled_stype
    is_dominant_soil_and_surface_type = is_dominant_surface_type & is_dominant_stype

    area_mean_over_dominant_sfc_and_stype = weighted_block_average(
        ds[AREA_MEAN_OVER_DOMINANT_SFC_AND_STYPE].where(
            is_dominant_soil_and_surface_type
        ),
        area.where(is_dominant_soil_and_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )

    area_mean = weighted_block_average(
        ds[AREA_MEAN], area, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_1"
    )

    area_mean_over_dominant_sfc_type = weighted_block_average(
        ds[AREA_MEAN_OVER_DOMINANT_SFC_TYPE].where(is_dominant_surface_type),
        area.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )

    area_and_vfrac_mean_over_dominant_sfc_and_vtype = weighted_block_average(
        ds[AREA_AND_VFRAC_MEAN_OVER_DOMINANT_SFC_AND_VTYPE].where(
            is_dominant_vegetation_and_surface_type
        ),
        (area * ds.vfrac).where(is_dominant_vegetation_and_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )

    area_mean_over_dominant_sfc_and_vtype = weighted_block_average(
        ds[AREA_AND_VFRAC_MEAN_OVER_DOMINANT_SFC_AND_VTYPE].where(
            is_dominant_vegetation_and_surface_type
        ),
        area.where(is_dominant_vegetation_and_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )

    coarsened_area_times_vfrac = block_coarsen(
        (ds.vfrac * area).where(is_dominant_vegetation_and_surface_type),
        coarsening_factor,
    )
    zorl_and_canopy = xr.where(
        coarsened_area_times_vfrac > 0.0,
        area_and_vfrac_mean_over_dominant_sfc_and_vtype,
        area_mean_over_dominant_sfc_and_vtype,
    )

    area_and_sncovr_mean = weighted_block_average(
        ds[AREA_AND_SNCOVR_MEAN],
        area * ds.sncovr,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    ).fillna(
        0.0
    )  # Any spots with NaN's are a result of zero snow cover.

    area_and_fice_mean = weighted_block_average(
        ds[AREA_AND_FICE_MEAN],
        area * ds.fice,
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    ).fillna(
        0.0
    )  # Any spots with NaN's are a result of zero ice cover.

    minimum_over_dominant_sfc_type = block_coarsen(
        ds[MINIMUM_OVER_DOMINANT_SFC_TYPE].where(is_dominant_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
        method="min",
    )

    maximum_over_dominant_sfc_type = block_coarsen(
        ds[MAXIMUM_OVER_DOMINANT_SFC_TYPE].where(is_dominant_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
        method="max",
    )

    # Handle tisfc specially
    tisfc_sea_ice = weighted_block_average(
        ds.tisfc.where(is_dominant_surface_type),
        (area * ds.fice).where(is_dominant_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )
    tisfc_land_or_ocean = weighted_block_average(
        ds.tisfc.where(is_dominant_surface_type),
        area.where(is_dominant_surface_type),
        coarsening_factor,
        x_dim="xaxis_1",
        y_dim="yaxis_1",
    )
    tisfc = xr.where(mode.slmsk == 2.0, tisfc_sea_ice, tisfc_land_or_ocean).rename(
        "tisfc"
    )

    result = xr.merge(
        [
            mode,
            tisfc,
            mode_over_dominant_surface_type,
            area_mean_over_dominant_sfc_and_stype,
            zorl_and_canopy,
            area_mean,
            area_mean_over_dominant_sfc_type,
            area_and_sncovr_mean,
            area_and_fice_mean,
            minimum_over_dominant_sfc_type,
            maximum_over_dominant_sfc_type,
        ]
    )

    result = _apply_surface_chgres_corrections(result)
    return doubles_to_floats(result)


def coarse_grain_grid_spec(
    ds,
    coarsening_factor,
    x_dim_unstaggered="grid_xt",
    y_dim_unstaggered="grid_yt",
    x_dim_staggered="grid_x",
    y_dim_staggered="grid_y",
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
    x_dim_unstaggered="grid_xt",
    y_dim_unstaggered="grid_yt",
    x_dim_staggered="grid_x",
    y_dim_staggered="grid_y",
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
            grid_spec.area.rename({"grid_xt": "xaxis_1", "grid_yt": "yaxis_2"}),
            grid_spec.dx.rename({"grid_xt": "xaxis_1", "grid_y": "yaxis_1"}),
            grid_spec.dy.rename({"grid_x": "xaxis_2", "grid_yt": "yaxis_2"}),
            coarsening_factor,
        )
    elif category == "fv_srf_wnd_coarse.res":
        coarsened = coarse_grain_fv_srf_wnd(
            source,
            grid_spec.area.rename({"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}),
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
            fv_core.delp.rename({"yaxis_2": "yaxis_1"}),
            grid_spec.area.rename({"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}),
            coarsening_factor,
        )
    elif category == "sfc_data":
        native_grid_spec = xr.open_mfdataset(native_grid_spec, concat_dim=tile)
        coarsened = coarse_grain_sfc_data(
            source,
            native_grid_spec.area.rename({"grid_xt": "xaxis_1", "grid_yt": "yaxis_1"}),
            coarsening_factor,
        )
    else:
        raise ValueError(
            f"Cannot coarse grain files for unknown 'category'," "{category}."
        )

    coarsened = sync_dimension_order(coarsened, source)
    for tile, file in zip(TILES, output_files):
        coarsened.sel(tile=tile).drop("tile").to_netcdf(file)
