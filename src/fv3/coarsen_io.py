import os

import pandas as pd
import xarray as xr

from ..data.cubedsphere import open_cubed_sphere
from .coarsen import (
    coarse_grain_grid_spec,
    coarse_grain_fv_core,
    coarse_grain_fv_srf_wnd,
    coarse_grain_fv_tracer,
    coarse_grain_sfc_data
)

TILES = range(1, 7)
OUTPUT_CATEGORY_NAMES = {
    'fv_core.res': 'fv_core_coarse.res',
    'fv_srf_wnd.res': 'fv_srf_wnd_coarse.res',
    'fv_tracer.res': 'fv_tracer_coarse.res',
    'sfc_data': 'sfc_data'
}
SOURCE_DATA_PATTERN = '{timestep}/{timestep}.{category}'


def sync_dimension_order(a, b):
    for var in a:
        a[var] = a[var].transpose(*b[var].dims)
    return a


def coarsen_grid_spec(
    input_grid_spec,
    target_resolution,
    output_filename,
    x_dim_unstaggered='grid_xt',
    y_dim_unstaggered='grid_yt',
    x_dim_staggered='grid_x',
    y_dim_staggered='grid_y'
):
    tile = pd.Index(TILES, name='tile')
    native_grid_spec = xr.open_mfdataset(input_grid_spec, concat_dim=tile)
    result = coarse_grain_grid_spec(
        native_grid_spec,
        target_resolution,
        x_dim_unstaggered,
        y_dim_unstaggered,
        x_dim_staggered,
        y_dim_staggered
    )
    result.to_netcdf(output_filename)


def coarsen_restart_file_category(
        timestep,
        native_category_name,
        target_resolution,
        coarse_grid_spec,
        native_grid_spec,
        source_data_prefix,
        output_files,
        source_data_pattern=SOURCE_DATA_PATTERN,
):
    category = OUTPUT_CATEGORY_NAMES[native_category_name]
    grid_spec = xr.open_dataset(coarse_grid_spec, chunks={'tile': 1})
    tile = pd.Index(TILES, name='tile')
    source = open_cubed_sphere(
        os.path.join(
            source_data_prefix,
            source_data_pattern.format(timestep=timestep, category=category)
        )
    )

    if category == 'fv_core_coarse.res':
        coarsened = coarse_grain_fv_core(
            source,
            source.delp,
            grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_2'}),
            grid_spec.dx.rename({'grid_xt': 'xaxis_1', 'grid_y': 'yaxis_1'}),
            grid_spec.dy.rename({'grid_x': 'xaxis_2', 'grid_yt': 'yaxis_2'}), 
            target_resolution
        )
    elif category == 'fv_srf_wnd_coarse.res':
        coarsened = coarse_grain_fv_srf_wnd(
            source, 
            grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
            target_resolution
        )
    elif category == 'fv_tracer_coarse.res':
        fv_core = open_cubed_sphere(
            os.path.join(
                source_data_prefix,
                source_data_pattern.format(timestep=timestep, category='fv_core_coarse.res')
            )
        )
        coarsened = coarse_grain_fv_tracer(
            source,
            fv_core.delp.rename({'yaxis_2': 'yaxis_1'}),
            grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
            target_resolution
        )
    elif category == 'sfc_data':
        native_grid_spec = xr.open_mfdataset(native_grid_spec, concat_dim=tile)
        coarsened = coarse_grain_sfc_data(
            source, 
            native_grid_spec.area.rename({'grid_xt': 'xaxis_1', 'grid_yt': 'yaxis_1'}),
            target_resolution
        )
    else:
        raise ValueError(
            f"Cannot coarse grain files for unknown 'category',"
            "{category}."
        )

    coarsened = sync_dimension_order(coarsened, source)
    for tile, file in zip(TILES, output_files):
        coarsened.sel(tile=tile).drop('tile').to_netcdf(file)
