from vcm.calc.calc import local_time
from vcm.visualize import plot_cube, mappable_var
from vcm.visualize.plot_diagnostics import plot_diurnal_cycle
from fv3net.diagnostics.data import (
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
    DIURNAL_VAR_MAPPING,
    DQ_MAPPING,
    DQ_PROFILE_MAPPING,
    GLOBAL_2D_MAPS,
    MAPPABLE_VAR_KWARGS
)
from scipy.stats import binned_statistic_2d
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import os
from typing import Mapping
import logging
import sys

logger = logging.getLogger("one_step_diags")

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
    stride = 7
    
    # make 2-d var global mean time series
    
    section_name = '2-d var global mean time series'
    logger.info(f'Plotting {section_name}')
    
    global_mean_time_series_plots = []
    for var, specs in GLOBAL_MEAN_2D_VARS.items():
        for vartype, scale in zip(specs[VAR_TYPE_DIM], specs['scale']):
            f = plot_global_mean_time_series(
                states_and_tendencies.sel({VAR_TYPE_DIM: vartype})[var + "_global_mean"],
                states_and_tendencies.sel({VAR_TYPE_DIM: vartype})[var + "_global_mean_std"],
                vartype = vartype,
                scale = scale
            )
            plotname = f"{var}_{vartype}_global_mean_time_series.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            global_mean_time_series_plots.append(plotname)
    report_sections[section_name] = global_mean_time_series_plots
    
    
    # make diurnal cycle comparisons between datasets
    
    section_name = 'diurnal cycle comparisons'
    logger.info(f'Plotting {section_name}')
    
    averages = ['land', 'sea']

    diurnal_cycle_comparisons = []
    for var, spec in DIURNAL_VAR_MAPPING.items():
        scale = spec['scale']
        for domain in averages:
            f = plot_diurnal_cycles(states_and_tendencies, '_'.join([var, domain]), stride = stride, scale = scale)
            plotname = f"{var}_{domain}.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            diurnal_cycle_comparisons.append(plotname)
    report_sections[section_name] = diurnal_cycle_comparisons
    
    
    # compare dQ with hi-res diagnostics
    
    section_name = 'dQ vs hi-res diagnostics across forecast time'
    logger.info(f'Plotting {section_name}')
    
    dQ_comparison_maps = []
    for q_term, mapping in DQ_MAPPING.items():
        coarse_var = mapping['coarse_name']
        hi_res_diag_var = mapping['hi-res_name']
        scale = mapping['scale']
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
                .rename({coarse_var: hi_res_diag_var})
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
            'states',
            FORECAST_TIME_DIM,
            stride = stride,
            scale = scale
        )
        plotname = f"{q_term}_comparison_maps.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        dQ_comparison_maps.append(plotname)
    report_sections[section_name] = dQ_comparison_maps
    
    
    # make vertical profiles of dQ terms across forecast time
    
    section_name = 'dQ profiles across forecast time'
    logger.info(f'Plotting {section_name}')
    
    composites = ['pos_PminusE_land_mean', 'neg_PminusE_land_mean', 'pos_PminusE_sea_mean', 'neg_PminusE_sea_mean', ]
    dQ_profile_maps = []
    for ds_name, dQ_info in DQ_PROFILE_MAPPING.items():
        dQ_name = dQ_info['name']
        dQ_type = dQ_info[VAR_TYPE_DIM]
        scale =  dQ_info["scale"]
        f = plot_dQ_vertical_profiles(
            states_and_tendencies.sel({VAR_TYPE_DIM: dQ_type, DELTA_DIM: 'hi-res - coarse'}),
            ds_name,
            dQ_name,
            dQ_type,
            composites,
            stride = stride,
            scale = scale
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
    for var, spec in GLOBAL_MEAN_3D_VARS.items():
        vartype = spec[VAR_TYPE_DIM]
        scale = spec["scale"]
        for average in averages:
            f, _ = plot_mean_time_height(
                states_and_tendencies.sel({VAR_TYPE_DIM: vartype})[f"{var}_{average}_mean"],
                vartype,
                scale
            )
            plotname = f"{var}_{vartype}_{average}_mean_time_height.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            mean_time_height_plots.append(plotname)
    report_sections[section_name] = mean_time_height_plots

    
    # make maps of 2-d vars across forecast time
    
    section_name = '2-d var maps across forecast time'
    logger.info(f'Plotting {section_name}')

    maps_across_forecast_time = []
    for var, spec in GLOBAL_2D_MAPS.items():
        vartype = spec[VAR_TYPE_DIM]
        scale = spec["scale"]
        f = plot_model_run_maps_across_time_dim(
            states_and_tendencies.sel({VAR_TYPE_DIM: vartype}),
            var,
            vartype,
            FORECAST_TIME_DIM,
            stride = stride,
            scale = scale
        )
        plotname = f"{var}_{vartype}_maps.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        maps_across_forecast_time.append(plotname)
    report_sections[section_name] = maps_across_forecast_time
    
    
    return report_sections


def plot_global_mean_time_series(
    da_mean: xr.DataArray,
    da_std:  xr.DataArray,
    vartype: str = 'tendency',
    scale: float = None
) -> plt.figure:
    
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
    if vartype == 'tendencies':
        ax.plot([da_mean[FORECAST_TIME_DIM].values[0], da_mean[FORECAST_TIME_DIM].values[-1]], [0, 0], 'k-')
        da_mean.attrs['units'] = da_mean.attrs['units'] + '/s'
    ax.set_xlabel(f"{FORECAST_TIME_DIM} [m]")
    ax.set_xlim([da_mean[FORECAST_TIME_DIM].values[0], da_mean[FORECAST_TIME_DIM].values[-1]])
    ax.set_xticks(da_mean[FORECAST_TIME_DIM])
    ax.set_ylabel(f"{da_mean.name} [{da_mean.attrs.get('units', None)}]")
    if scale is not None:
        ax.set_ylim([-scale, scale])
    ax.set_title(f"{da_mean.name} {vartype}")
    f.set_size_inches([10, 4])
    f.set_dpi(FIG_DPI)
    
    return f


def plot_model_run_maps_across_time_dim(
    ds: xr.Dataset,
    var: str,
    vartype: str,
    multiple_time_dim: str,
    start: int = None,
    end: int = None,
    stride: int = None,
    scale: float = None
):
    
    rename_dims = {'x': 'grid_xt', 'y': 'grid_yt', 'x_interface': 'grid_x', 'y_interface': 'grid_y'}
    if vartype == "tendencies":
        ds[var].attrs.update({'units': ds[var].attrs['units'] + '/s'})
    ds = ds.assign_coords({FORECAST_TIME_DIM: ds[FORECAST_TIME_DIM]/60})
    f, axes, _, _, facet_grid = plot_cube(
        mappable_var(ds.isel({multiple_time_dim: slice(start, end, stride)}), var, **MAPPABLE_VAR_KWARGS),
        col = DELTA_DIM,
        row = multiple_time_dim,
        vmax = scale
    )
    n_rows = ds.isel({multiple_time_dim: slice(start, end, stride)}).sizes[multiple_time_dim]
    f.set_size_inches([10, n_rows*2])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{var} across {multiple_time_dim}")
    facet_grid.set_titles(template='{value}', maxchar=30)
    # add units to facetgrid right side (hacky)
    if multiple_time_dim == FORECAST_TIME_DIM:
        right_text_objs = [vars(ax)['texts'][0] for ax in axes[:, -1]]
        for obj in right_text_objs:
            right_text = obj.get_text()
            obj.set_text(right_text + ' min')
    
    
    return f


def plot_mean_time_height(
    th_da: xr.DataArray,
    vartype: str,
    scale: float = None
) -> plt.figure:
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
        vmax = scale,
    )    
    plt.suptitle(f"{th_da.attrs['long_name']} across {FORECAST_TIME_DIM}")
    ex_ax = facetgrid.axes.flatten()[0]
    ex_ax.set_xlabel(f"{FORECAST_TIME_DIM} [minutes]")
    ex_ax.set_xlim([
        th_da[FORECAST_TIME_DIM].values[0] - 0.5,
        th_da[FORECAST_TIME_DIM].values[-1] + 0.5
    ])
    ex_ax.set_xticks(th_da[FORECAST_TIME_DIM])
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
    stride: int = None,
    scale: float = None,
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
    if scale is not None:
        facetgrid.axes.flatten()[0].set_xlim([-scale, scale])
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
    f.set_size_inches([14, n_rows*4])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{dQ_name}")
    
    return facetgrid.fig


def plot_diurnal_cycles(
    ds: xr.Dataset,
    var: str,
    start: int = None,
    end: int = None,
    stride: int = None,
    scale: float = None
) -> plt.figure:
    
    ds = ds.assign_coords({FORECAST_TIME_DIM: (ds[FORECAST_TIME_DIM]/60).astype(int)})

    def _facet_line_plot(arr: np.ndarray):
        ax = plt.gca()
        ax.set_prop_cycle(color=['b', [1, 0.5, 0]])
        h = ax.plot(arr.T)
        ax.plot([0, 24.], [0, 0], 'k-')
        return h
    
    facetgrid = xr.plot.FacetGrid(
        data = ds.isel({FORECAST_TIME_DIM: slice(start, end, stride), DELTA_DIM: slice(None, 2)}),
        col = FORECAST_TIME_DIM,
        col_wrap = 4
    )
    
    facetgrid = facetgrid.map(_facet_line_plot, var)
    facetgrid.axes.flatten()[0].set_xlim([0, 24])
    legend_ax = facetgrid.axes.flatten()[-2]
    handles = legend_ax.get_lines()
    legend_ax.legend(handles, ['hi-res diags', 'coarse tendencies'], loc=2)
    facetgrid.set_titles(template='{value} minutes')
    for ax in facetgrid.axes[-1, :]:
        ax.set_xlabel("mean local time [hrs]")
        ax.set_xticks(np.arange(0., 24., 4.))
    for ax in facetgrid.axes[:, 0]:
        ax.set_ylabel(f"{var} [{ds[var].attrs['units']}]")
    if scale is not None:
        ax.set_ylim([-scale, scale])
    n_rows = facetgrid.axes.shape[0]
    f = facetgrid.fig
    f.set_size_inches([12, n_rows*4])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{var}")
    
    return facetgrid.fig