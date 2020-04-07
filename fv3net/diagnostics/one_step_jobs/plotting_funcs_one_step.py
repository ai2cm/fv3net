from vcm.calc.calc import local_time
from vcm.visualize import plot_cube, mappable_var
from vcm.visualize.plot_diagnostics import plot_diurnal_cycle
from fv3net.diagnostics.data_funcs import (
    get_latlon_grid_coords_set,
    EXAMPLE_CLIMATE_LATLON_COORDS,
)
from fv3net.diagnostics.one_step_jobs import (
    FORECAST_TIME_DIM,
    VAR_TYPE_DIM,
    DELTA_DIM,
    GRID_VARS,
    ABS_VARS,
    GLOBAL_MEAN_2D_VARS,
    GLOBAL_MEAN_3D_VARS,
)
from scipy.stats import binned_statistic_2d
import xarray as xr
from matplotlib import pyplot as plt
import os
from typing import Mapping


FIG_DPI = 100

def make_all_plots(states_and_tendencies: xr.Dataset, output_dir: str) -> Mapping:
    """ Makes figures for predictions on test data

    Args:
        states_and_tendencies: processed dataset of outputs from one-step
            jobs, containing states and tendencies of both the hi-res and
            coarse model, averaged across initial times for various 2-D,
            variables, global/land/sea mean time series, and global/land/sea
            mean time-height series, i=output from
            fv3net.diagnostics.one_step_jobs
        output_dir: location to write figures to

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """
    
    report_sections = {}
    
    # make 2-d var global mean time series
    global_mean_time_series_plots = []
    for var in GLOBAL_MEAN_2D_VARS:
        for vartype in ['tendencies', 'states']:
            f = plot_global_mean_time_series(
                states_and_tendencies.sel(var_type = vartype)[var + "_global_mean"],
                states_and_tendencies.sel(var_type = vartype)[var + "_global_mean_std"],
                vartype
            )
            plotname = f"{var}_{vartype}_global_mean_time_series.png"
            f.savefig(os.path.join(output_dir, plotname))
            global_mean_time_series_plots.append(plotname)
    report_sections['2-d var global mean time series'] = global_mean_time_series_plots
    
    # make maps of 2-d vars across forecast time
    
    maps_to_make = {
        "tendencies": ['column_integrated_heating'],
        "states":  []
    }
    i_start = None
    i_end = None
    stride = 2

    maps_across_forecast_time = []
    for vartype, var_list in maps_to_make.items():
        for var in var_list:
            f = plot_model_run_maps_across_time_dim(
                states_and_tendencies.sel({VAR_TYPE_DIM: vartype}),
                var,
                FORECAST_TIME_DIM,
                stride = stride
            )
            plotname = f"{var}_{vartype}_maps.png"
            f.savefig(os.path.join(output_dir, plotname))
            maps_across_forecast_time.append(plotname)
    report_sections['2-d var maps across forecast time'] = maps_across_forecast_time
    
    # compare dQ with hi-res diagnostics
    
    dQ_mapping = {
        'Q1': {
            'column_integrated_heating': 'net_heating'
        },
        'Q2': {
            'column_integrated_moistening': 'net_precipitation'
        }
    }
    
    dQ_comparison_maps = []
    for q_term, mapping in dQ_mapping.items():
        for coarse_var, hi_res_diag_var in mapping.items():
            hi_res_diag_var_full = hi_res_diag_var + "_physics"
            comparison_ds = xr.concat([
                (
                    states_and_tendencies[list((hi_res_diag_var_full,) + GRID_VARS)]
                    .sel({DELTA_DIM: 'hi-res', VAR_TYPE_DIM: 'states'})
                    .drop([DELTA_DIM, VAR_TYPE_DIM])
                    .expand_dims({DELTA_DIM: ['hi-res diagnostics']})
                    .rename({hi_res_diag_var_full: hi_res_diag_var})
                ),
                (
                    states_and_tendencies[list((coarse_var,) + GRID_VARS)]
                    .sel({DELTA_DIM: 'hi-res - coarse', VAR_TYPE_DIM: 'tendencies'})
                    .drop([DELTA_DIM, VAR_TYPE_DIM])
                    .expand_dims({DELTA_DIM: ['tendency-based dQ']})
                    .rename(mapping)
                )
            ], dim=DELTA_DIM
            )
            comparison_ds[hi_res_diag_var] = (
                comparison_ds[hi_res_diag_var]
                .assign_attrs({"long_name": hi_res_diag_var, "units": "mm/day"})
            )
            f = plot_model_run_maps_across_time_dim(
                comparison_ds,
                hi_res_diag_var,
                FORECAST_TIME_DIM,
                stride = stride)
            plotname = f"{hi_res_diag_var}_comparison_mapss.png"
            f.savefig(os.path.join(output_dir, plotname))
            dQ_comparison_maps.append(plotname)
    report_sections['dQ vs hi-res diagnostics across forecast time'] = dQ_comparison_maps
    
    return report_sections


def plot_global_mean_time_series(da_mean: xr.DataArray, da_std:  xr.DataArray, vartype: str = 'tendency') -> plt.figure:
    
    init_color1 = [0.75, 0.75, 1]
    init_color2 = [0.5, 0.75, 0.5]
    da_mean = da_mean.assign_coords({FORECAST_TIME_DIM: da_mean[FORECAST_TIME_DIM]/60})
    da_std = da_std.assign_coords({FORECAST_TIME_DIM: da_std[FORECAST_TIME_DIM]/60})
    f = plt.figure()
    ax = plt.subplot()
    h1 = ax.fill_between(
        da_mean[FORECAST_TIME_DIM],
        (da_mean + 1.96*da_std).sel(model_run='coarse'),
        (da_mean - 1.96*da_std).sel(model_run='coarse'),
        color=init_color1,
        alpha = 0.5
    )
    h2, = ax.plot(da_mean[FORECAST_TIME_DIM], da_mean.sel(model_run='coarse'), 'b-o')
    h3 = ax.fill_between(
        da_mean[FORECAST_TIME_DIM],
        (da_mean + 1.96*da_std).sel(model_run='hi-res'),
        (da_mean - 1.96*da_std).sel(model_run='hi-res'),
        color=init_color2,
        alpha = 0.5
    )
    h4, = ax.plot(da_mean[FORECAST_TIME_DIM], da_mean.sel(model_run='hi-res'), 'g-x')
    handles = [h1, h2, h3, h4]
    ax.legend(handles, ['Coarse initializations', 'Coarse mean', 'Hi-res initializations', 'Hi-res mean'])
    if vartype == 'tendency':
        ax.plot([da_mean[FORECAST_TIME_DIM].values[0], da_mean[FORECAST_TIME_DIM].values[-1]], [0, 0], 'k-')
    ax.set_xlabel(f"{FORECAST_TIME_DIM} [m]")
    ax.set_xlim([da_mean[FORECAST_TIME_DIM].values[0], da_mean[FORECAST_TIME_DIM].values[-1]])
    ax.set_xticks(da_mean[FORECAST_TIME_DIM])
    ax.set_ylabel(f"{da_mean.attrs['long_name']} [{da_mean.attrs.get('units')}]")
    ax.set_title(f"{da_mean.name} {vartype}")
    f.set_size_inches([10, 4])
    f.set_dpi(FIG_DPI)
    
    return f


def plot_model_run_maps_across_time_dim(ds, var, multiple_time_dim, start = None, end = None, stride = None):
    
    rename_dims = {'x': 'grid_xt', 'y': 'grid_yt', 'x_interface': 'grid_x', 'y_interface': 'grid_y'}
    ds = ds.assign_coords({FORECAST_TIME_DIM: ds[FORECAST_TIME_DIM]/60})
    f, axes, _, _, facet_grid = plot_cube(
        mappable_var(ds.isel({multiple_time_dim: slice(start, end, stride)}).rename(rename_dims), var),
        col = DELTA_DIM,
        row = multiple_time_dim,
        cmap_percentiles_lim=False
    )
    n_rows = ds.isel({multiple_time_dim: slice(start, end, stride)}).sizes[multiple_time_dim]
    f.set_size_inches([10, n_rows*2])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{ds[var].attrs['long_name']} across {multiple_time_dim}")
    facet_grid.set_titles(template='{value}', maxchar=30)
    # add units to facetgrid right side (hacky)
    if multiple_time_dim == FORECAST_TIME_DIM:
        right_text_objs = [vars(ax)['texts'][0] for ax in axes[:, -1]]
        for obj in right_text_objs:
            right_text = obj.get_text()
            obj.set_text(right_text + ' min')
    
    return f