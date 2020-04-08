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
import numpy as np
from matplotlib import pyplot as plt
import os
from typing import Mapping
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
logger.addHandler(out_hdlr)


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
    
    
    # compare dQ with hi-res diagnostics
    
    section_name = 'dQ vs hi-res diagnostics across forecast time'
    logger.info(f'Plotting {section_name}')
    
    dQ_mapping = {
        'Q1': {
            'column_integrated_heating': 'net_heating'
        },
        'Q2': {
            'column_integrated_moistening': 'net_precipitation'
        }
    }
    stride = 2
    
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
                .assign_attrs({"long_name": hi_res_diag_var})
            )
            f = plot_model_run_maps_across_time_dim(
                comparison_ds,
                hi_res_diag_var,
                FORECAST_TIME_DIM,
                stride = stride)
            plotname = f"{hi_res_diag_var}_comparison_maps.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            dQ_comparison_maps.append(plotname)
    report_sections[section_name] = dQ_comparison_maps
    
    
    # make vertical profiles of dQ terms across forecast time
    
    section_name = 'dQ profiles across forecast time'
    logger.info(f'Plotting {section_name}')
    
    dQ_PROFILE_MAPPING = {
        'air_temperature': {
            'name': 'dQ1',
            VAR_TYPE_DIM: 'tendencies'
        },
        'specific_humidity': {
            'name': 'dQ2',
            VAR_TYPE_DIM: 'tendencies'
        },
        'vertical_wind': {
            'name': 'dW',
            VAR_TYPE_DIM: 'states'
        },
    }
    
    composites = ['pos_PminusE_land_mean', 'neg_PminusE_land_mean', 'pos_PminusE_sea_mean', 'neg_PminusE_sea_mean', ]
    dQ_profile_maps = []
    for ds_name, dQ_info in dQ_PROFILE_MAPPING.items():
        dQ_name = dQ_info['name']
        dQ_type = dQ_info[VAR_TYPE_DIM]
        f = plot_dQ_vertical_profiles(
            states_and_tendencies.sel({VAR_TYPE_DIM: dQ_type, DELTA_DIM: 'hi-res - coarse'}),
            ds_name,
            dQ_name,
            dQ_type,
            composites,
            stride = 2
        )
        plotname = f"{dQ_name}_profiles.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        dQ_profile_maps.append(plotname)
    report_sections[section_name] = dQ_profile_maps

    
    # make time-height plots of mean 3-D variables
    
    section_name = '3-d var mean time height'
    logger.info(f'Plotting {section_name}')
    
    averages = ['global', 'sea', 'land']

    mean_time_height_plots = []
    for var in GLOBAL_MEAN_3D_VARS:
        for vartype in ['states', 'tendencies']:
            for average in averages:
                f, _ = plot_mean_time_height(
                    states_and_tendencies.sel({VAR_TYPE_DIM: vartype})[f"{var}_{average}_mean"],
                    vartype
                )
                plotname = f"{var}_{vartype}_{average}_mean_time_height.png"
                f.savefig(os.path.join(output_dir, plotname))
                plt.close(f)
                mean_time_height_plots.append(plotname)
    report_sections[section_name] = mean_time_height_plots
    
    
    # make 2-d var global mean time series
    
    section_name = '2-d var global mean time series'
    logger.info(f'Plotting {section_name}')
    
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
            plt.close(f)
            global_mean_time_series_plots.append(plotname)
    report_sections[section_name] = global_mean_time_series_plots
    
    
    # make maps of 2-d vars across forecast time
    
    section_name = '2-d var maps across forecast time'
    logger.info(f'Plotting {section_name}')
    
    maps_to_make = {
        "tendencies": ['psurf', 'column_integrated_heating', 'column_integrated_moistening'],
        "states":  ['vertical_wind_level_40']
    }

    maps_across_forecast_time = []
    for vartype, var_list in maps_to_make.items():
        for var in var_list:
            for subvar in [var, f"{var}_std"]:
                f = plot_model_run_maps_across_time_dim(
                    states_and_tendencies.sel({VAR_TYPE_DIM: vartype}),
                    subvar,
                    FORECAST_TIME_DIM,
                    stride = stride
                )
                plotname = f"{subvar}_{vartype}_maps.png"
                f.savefig(os.path.join(output_dir, plotname))
                plt.close(f)
                maps_across_forecast_time.append(plotname)
    report_sections[section_name] = maps_across_forecast_time
    
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
        cmap_percentiles_lim=True
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


def plot_mean_time_height(th_da: xr.DataArray, vartype: str) -> plt.figure:
    th_da = th_da.assign_coords({FORECAST_TIME_DIM: th_da[FORECAST_TIME_DIM]/60})
    if vartype == 'tendencies':
        if 'long_name' in th_da.attrs:
            th_da.attrs.update({
                'long_name' : f"{th_da.attrs['long_name']} tendency",
                'units' : f"{th_da.attrs['units']}/s"
            })
        else:
            th_da.attrs.update({
                'long_name' : f"{th_da.name} tendency",
                'units' : f"{th_da.attrs['units']}/s"
            })
    elif 'long_name' not in th_da.attrs:
        th_da.attrs.update({
            'long_name' : f"{th_da.name}"
        })
    facetgrid = th_da.plot(
        x=FORECAST_TIME_DIM,
        y='z',
        col=DELTA_DIM,
        yincrease=False,
    )    
    plt.suptitle(f"{th_da.attrs['long_name']} across {FORECAST_TIME_DIM}")
    for ax in facetgrid.axes.flatten():
        ax.set_xlabel(f"{FORECAST_TIME_DIM} [minutes]")
        ax.set_xlim([
            th_da[FORECAST_TIME_DIM].values[0] - 0.5,
            th_da[FORECAST_TIME_DIM].values[-1] + 0.5
        ])
        ax.set_xticks(th_da[FORECAST_TIME_DIM])
    f = facetgrid.fig
    f.set_dpi(FIG_DPI)
    f.set_size_inches([12, 5])
    
    return f, facetgrid


def plot_dQ_vertical_profiles(
    ds: xr.DataArray,
    ds_name: str,
    dQ_name: str,
    dQ_type: str,
    composites: list,
    start: int = None,
    end: int = None,
    stride: int = None
) -> plt.figure:
    
    ds = ds.assign_coords({FORECAST_TIME_DIM: (ds[FORECAST_TIME_DIM]/60).astype(int)})
    varnames = ['_'.join([ds_name, suffix]) for suffix in composites]
    ds_across_vars = xr.Dataset({dQ_name: xr.concat([
        ds[var].expand_dims({'dQ_composite': [suffix]}) for var, suffix in zip(varnames, composites)
    ], dim='dQ_composite')})
    
    if dQ_type == 'tendencies':
        ds_across_vars[dQ_name] = ds_across_vars[dQ_name].assign_attrs({
            'long_name': dQ_name + ' tendency',
            'units': ds[varnames[0]].attrs['units'] + '/s'
        })

    def _facet_line_plot(arr: np.ndarray):
        ax = plt.gca()
        ax.set_prop_cycle(color=['b', [1, 0.5, 0], 'b', [1, 0.5, 0]],
                          linestyle=['-', '-', '--', '--']
                         )
        nz = arr.shape[1] + 1.
        h = ax.plot(arr.T, np.arange(1., nz))
        ax.plot([0, 0], [1, nz], 'k--')
        return h
    
    facetgrid = xr.plot.FacetGrid(
        data=ds_across_vars.isel({FORECAST_TIME_DIM: slice(start, end, stride)}),
        col = FORECAST_TIME_DIM,
        col_wrap = 4
    )
    
    facetgrid = facetgrid.map(_facet_line_plot, dQ_name)
    facetgrid.axes.flatten()[0].invert_yaxis()
    facetgrid.axes.flatten()[0].set_ylim([ds.sizes['z'], 1])
    legend_ax = facetgrid.axes.flatten()[-2]
    handles = legend_ax.get_lines()
    legend_ax.legend(handles[:-1], composites, loc=2)
    facetgrid.set_titles(template='{value} minutes')
    for ax in facetgrid.axes[-1, :]:
        ax.set_xlabel(f"{dQ_name} [{ds_across_vars[dQ_name].attrs['units']}]")
    for ax in facetgrid.axes[:, 0]:
        ax.set_ylabel('model level')
    n_rows = facetgrid.axes.shape[0]
    f = facetgrid.fig
    f.set_size_inches([12, n_rows*4])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{dQ_name}")
    
    return facetgrid.fig