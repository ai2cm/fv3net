from vcm.visualize import plot_cube, mappable_var
from .constants import FORECAST_TIME_DIM, DELTA_DIM, INIT_TIME_DIM
from .config import MAPPABLE_VAR_KWARGS, GRID_VARS
import gallery
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
import os
from typing import Mapping
import logging

logger = logging.getLogger("one_step_diags")

FIG_DPI = 100
TIME_FMT = "%Y%m%d.%H%M%S"


def make_all_plots(
    states_and_tendencies: xr.Dataset,
    config: Mapping,
    output_dir: str,
    stride: int = 7,
    time1: int = 7,
    time2: int = 14,
) -> Mapping:
    """ Makes figures for one-step diagnostics

    Args:
        states_and_tendencies: processed dataset of outputs from one-step
            jobs, containing states and tendencies of both the hi-res and
            coarse model, averaged across initial times for various 2-D,
            variables, global/land/sea mean time series, and global/land/sea
            mean time-height series, output from
            fv3net.diagnostics.one_step_jobs
        config: diagnostics configuration dict specifying plots to make
        output_dir: location for writing figures
        stride: int stride for subplotting along forecast time dimension;
            default 7
        time1: int index of first forecast time to plot in 2x2 map plot format;
            default 7
        time2: int index of second forecast time to plot in 2x2 map plot format;
            default 14
        

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """

    report_sections = {}

    # timesteps dist

    timestep_strings = (states_and_tendencies.attrs[INIT_TIME_DIM]).split(" ")
    timesteps = [
        datetime.strptime(timestep_string, TIME_FMT)
        for timestep_string in timestep_strings
    ]
    f = plot_timestep_counts(timesteps)
    plotname = "timesteps_distribution.png"
    f.savefig(os.path.join(output_dir, plotname))
    plt.close(f)
    report_sections["timestep distribution"] = [plotname]

    # make 2-d var global mean time series

    section_name = "2-d var global mean time series"
    logger.info(f"Plotting {section_name}")

    global_mean_time_series_plots = []
    for var, specs in config["GLOBAL_MEAN_2D_VARS"].items():
        for vartype, scale in zip(specs["var_type"], specs["scale"]):
            f = plot_global_mean_time_series(
                states_and_tendencies.sel({"var_type": vartype})[var + "_global_mean"],
                states_and_tendencies.sel({"var_type": vartype})[
                    var + "_global_mean_std"
                ],
                vartype=vartype,
                scale=scale,
            )
            plotname = f"{var}_{vartype}_global_mean_time_series.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            global_mean_time_series_plots.append(plotname)
    report_sections[section_name] = global_mean_time_series_plots

    # make diurnal cycle comparisons between datasets

    section_name = "diurnal cycle comparisons"
    logger.info(f"Plotting {section_name}")

    averages = ["global", "land", "sea"]

    diurnal_cycle_comparisons = []
    for var, spec in config["DIURNAL_VAR_MAPPING"].items():
        scale = spec["scale"]
        for domain in averages:
            f = plot_diurnal_cycles(
                states_and_tendencies,
                "_".join([var, domain]),
                stride=stride,
                scale=scale,
            )
            plotname = f"{var}_{domain}.png"
            f.savefig(os.path.join(output_dir, plotname))
            plt.close(f)
            diurnal_cycle_comparisons.append(plotname)
    report_sections[section_name] = diurnal_cycle_comparisons

    # compare dQ with hi-res diagnostics

    section_name = "dQ vs hi-res diagnostics across forecast time"
    logger.info(f"Plotting {section_name}")

    dQ_comparison_maps = []
    for q_term, mapping in config["DQ_MAPPING"].items():
        residual_var = mapping["tendency_diff_name"]
        physics_var = mapping["physics_name"]
        scale = mapping["scale"]
        physics_var_full = physics_var + "_physics"

        # manufacture a dataset with a "dimension" that's really a set of
        # diverse subpanel plots

        hi_res_time1 = (
            states_and_tendencies[physics_var_full]
            .sel({DELTA_DIM: "hi-res", "var_type": "states"})
            .isel({FORECAST_TIME_DIM: time1})
            .drop(DELTA_DIM)
            .rename(physics_var)
        )

        pQ_plus_dQ_time1 = (
            (
                states_and_tendencies[physics_var_full].sel(
                    {DELTA_DIM: "coarse", "var_type": "states"}
                )
                + states_and_tendencies[residual_var].sel(
                    {DELTA_DIM: "hi-res - coarse", "var_type": "tendencies"}
                )
            )
            .isel({FORECAST_TIME_DIM: time1})
            .rename(f"(pQ + dQ) at {time1} min")
        )

        hi_res_minus_pQ_plus_dQ_time1 = (
            (
                states_and_tendencies[physics_var_full].sel(
                    {DELTA_DIM: "hi-res", "var_type": "states"}
                )
                - states_and_tendencies[physics_var_full].sel(
                    {DELTA_DIM: "coarse", "var_type": "states"}
                )
                - states_and_tendencies[residual_var].sel(
                    {DELTA_DIM: "hi-res - coarse", "var_type": "tendencies"}
                )
            )
            .drop(DELTA_DIM)
            .isel({FORECAST_TIME_DIM: time1})
            .rename(f"hi-res physics - (pQ + dQ) at {time1} min")
        )

        pQ_plus_dQ_time2_minus_time1 = (
            states_and_tendencies[physics_var_full]
            .sel({DELTA_DIM: "coarse", "var_type": "states"})
            .isel({FORECAST_TIME_DIM: time2})
            + states_and_tendencies[residual_var]
            .sel({DELTA_DIM: "hi-res - coarse", "var_type": "tendencies"})
            .isel({FORECAST_TIME_DIM: time2})
            - states_and_tendencies[physics_var_full]
            .sel({DELTA_DIM: "coarse", "var_type": "states"})
            .isel({FORECAST_TIME_DIM: time1})
            - states_and_tendencies[residual_var]
            .sel({DELTA_DIM: "hi-res - coarse", "var_type": "tendencies"})
            .isel({FORECAST_TIME_DIM: time1})
        ).rename(f"(pQ + dQ) at {time2} min - at {time1} min")

        comparison_ds = xr.merge(
            [
                hi_res_time1,
                pQ_plus_dQ_time1,
                hi_res_minus_pQ_plus_dQ_time1,
                pQ_plus_dQ_time2_minus_time1,
            ]
        ).to_array(dim=DELTA_DIM, name=physics_var)
        comparison_ds = xr.merge(
            [comparison_ds, states_and_tendencies[list(GRID_VARS)]]
        )
        comparison_ds[physics_var] = comparison_ds[physics_var].assign_attrs(
            {
                "long_name": physics_var,
                "units": states_and_tendencies[physics_var_full].attrs.get(
                    "units", None
                ),
            }
        )

        f = plot_model_run_maps_across_forecast_time(
            comparison_ds, physics_var, DELTA_DIM, scale=scale
        )
        plotname = f"{q_term}_comparison_maps.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        dQ_comparison_maps.append(plotname)
    report_sections[section_name] = dQ_comparison_maps

    # make vertical profiles of dQ terms across forecast time

    section_name = "dQ profiles across forecast time"
    logger.info(f"Plotting {section_name}")

    dQ_profile_maps = []
    for ds_name, dQ_info in config["DQ_PROFILE_MAPPING"].items():
        dQ_name = dQ_info["name"]
        dQ_type = dQ_info["var_type"]
        scale = dQ_info["scale"]
        f = plot_dQ_vertical_profiles(
            states_and_tendencies.sel(
                {"var_type": dQ_type, DELTA_DIM: "hi-res - coarse"}
            ),
            ds_name,
            dQ_name,
            dQ_type,
            config["PROFILE_COMPOSITES"],
            stride=stride,
            scale=scale,
        )
        plotname = f"{dQ_name}_profiles.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        dQ_profile_maps.append(plotname)
    report_sections[section_name] = dQ_profile_maps

    # make time-height plots of mean 3-D variables

    section_name = "3-d var mean time height"
    logger.info(f"Plotting {section_name}")

    averages = ["global", "sea", "land"]

    mean_time_height_plots = []
    for var, spec in config["GLOBAL_MEAN_3D_VARS"].items():
        vartype = spec["var_type"]
        scale = spec["scale"]
        combined_ds = (
            xr.concat(
                [
                    (
                        states_and_tendencies.sel(
                            {"var_type": vartype, DELTA_DIM: "coarse"}
                        )[f"{var}_{average}_mean"]
                    )
                    for average in averages
                ],
                dim="composite",
            )
            .assign_coords({"composite": averages})
            .rename(f"{var} mean")
        )
        f, _ = plot_mean_time_height(combined_ds, vartype, scale,)
        plotname = f"{var}_{vartype}_mean_time_height.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        mean_time_height_plots.append(plotname)
    report_sections[section_name] = mean_time_height_plots

    # make maps of 2-d vars across forecast time

    section_name = "2-d var maps across forecast time"
    logger.info(f"Plotting {section_name}")

    maps_across_forecast_time = []
    for var, spec in config["GLOBAL_2D_MAPS"].items():
        vartype = spec["var_type"]
        scale = spec["scale"]

        # manufacture a dataset with a "dimension" that's really a set of
        # diverse subpanel plots

        hi_res_time1 = (
            states_and_tendencies[var]
            .sel({DELTA_DIM: "hi-res", "var_type": vartype})
            .isel({FORECAST_TIME_DIM: time1})
            .drop([DELTA_DIM, FORECAST_TIME_DIM])
            .rename(f"hi-res at {time1} min")
        )

        coarse_time1 = (
            states_and_tendencies[var]
            .sel({DELTA_DIM: "coarse", "var_type": vartype})
            .isel({FORECAST_TIME_DIM: time1})
            .drop([DELTA_DIM, FORECAST_TIME_DIM])
            .rename(f"coarse at {time1} min")
        )

        hi_res_minus_coarse_time1 = (
            (
                states_and_tendencies[var].sel(
                    {DELTA_DIM: "hi-res", "var_type": vartype}
                )
                - states_and_tendencies[var].sel(
                    {DELTA_DIM: "coarse", "var_type": vartype}
                )
            )
            .isel({FORECAST_TIME_DIM: time1})
            .drop(FORECAST_TIME_DIM)
            .rename(f"hi-res - coarse at {time1} min")
        )

        coarse_time2_minus_time1 = (
            (
                states_and_tendencies[var].isel({FORECAST_TIME_DIM: time2})
                - states_and_tendencies[var].isel({FORECAST_TIME_DIM: time1})
            )
            .sel({DELTA_DIM: "coarse", "var_type": vartype})
            .drop(DELTA_DIM)
            .rename(f"coarse at {time2} min - coarse at {time1} min")
        )

        comparison_ds = xr.merge(
            [
                hi_res_time1,
                coarse_time1,
                hi_res_minus_coarse_time1,
                coarse_time2_minus_time1,
            ]
        ).to_array(dim=DELTA_DIM, name=var)
        comparison_ds = xr.merge(
            [comparison_ds, states_and_tendencies[list(GRID_VARS)]]
        )
        comparison_ds[var].attrs.update(
            {"units": states_and_tendencies[var].attrs["units"]}
        )
        if vartype == "tendencies" and "column_integrated" not in var:
            if "long_name" in comparison_ds[var].attrs:
                comparison_ds[var].attrs.update(
                    {"long_name": comparison_ds[var].attrs["long_name"] + " tendency"}
                )
            comparison_ds[var].attrs.update(
                {"units": comparison_ds[var].attrs["units"] + "/s"}
            )
            comparison_ds = comparison_ds.rename({var: f"{var}_tendencies"})
            var = f"{var}_tendencies"

        f = plot_model_run_maps_across_forecast_time(
            comparison_ds, var, DELTA_DIM, scale=scale
        )
        plotname = f"{var}_maps.png"
        f.savefig(os.path.join(output_dir, plotname))
        plt.close(f)
        maps_across_forecast_time.append(plotname)

    report_sections[section_name] = maps_across_forecast_time

    return report_sections


def plot_global_mean_time_series(
    da_mean: xr.DataArray,
    da_std: xr.DataArray,
    vartype: str = "tendency",
    scale: float = None,
) -> plt.figure:

    init_color1 = [0.75, 0.75, 1]
    init_color2 = [0.5, 0.75, 0.5]
    da_mean = da_mean.assign_coords(
        {FORECAST_TIME_DIM: da_mean[FORECAST_TIME_DIM] / 60}
    )
    da_std = da_std.assign_coords({FORECAST_TIME_DIM: da_std[FORECAST_TIME_DIM] / 60})
    f = plt.figure()
    ax = plt.subplot()
    h1 = ax.fill_between(
        da_mean[FORECAST_TIME_DIM],
        (da_mean + 1.96 * da_std).sel(model_run="coarse"),
        (da_mean - 1.96 * da_std).sel(model_run="coarse"),
        color=init_color1,
        alpha=0.5,
    )
    (h2,) = ax.plot(da_mean[FORECAST_TIME_DIM], da_mean.sel(model_run="coarse"), "b-o")
    h3 = ax.fill_between(
        da_mean[FORECAST_TIME_DIM],
        (da_mean + 1.96 * da_std).sel(model_run="hi-res"),
        (da_mean - 1.96 * da_std).sel(model_run="hi-res"),
        color=init_color2,
        alpha=0.5,
    )
    (h4,) = ax.plot(da_mean[FORECAST_TIME_DIM], da_mean.sel(model_run="hi-res"), "g-x")
    handles = [h1, h2, h3, h4]
    ax.legend(
        handles,
        [
            "Coarse initializations",
            "Coarse mean",
            "Hi-res initializations",
            "Hi-res mean",
        ],
    )
    if vartype == "tendencies":
        ax.plot(
            [
                da_mean[FORECAST_TIME_DIM].values[0],
                da_mean[FORECAST_TIME_DIM].values[-1],
            ],
            [0, 0],
            "k-",
        )
        da_mean.attrs["units"] = da_mean.attrs["units"] + "/s"
    ax.set_xlabel(f"{FORECAST_TIME_DIM} [min]")
    ax.set_xlim(
        [da_mean[FORECAST_TIME_DIM].values[0], da_mean[FORECAST_TIME_DIM].values[-1]]
    )
    ax.set_xticks(da_mean[FORECAST_TIME_DIM])
    ax.set_ylabel(f"{da_mean.name} [{da_mean.attrs.get('units', None)}]")
    if scale is not None:
        if "abs" in da_mean.name:
            ax.set_ylim([0, scale])
        else:
            ax.set_ylim([-scale, scale])
    ax.set_title(f"{da_mean.name} {vartype}")
    f.set_size_inches([10, 4])
    f.set_dpi(FIG_DPI)

    return f


def plot_model_run_maps_across_forecast_time(
    ds: xr.Dataset, var: str, subpanel_dim: str, scale: float = None,
):

    f, axes, _, _, facet_grid = plot_cube(
        mappable_var(ds, var, **MAPPABLE_VAR_KWARGS),
        col=subpanel_dim,
        col_wrap=2,
        vmax=scale,
    )
    f.set_size_inches([10, 6])
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{var} across forecast time")
    facet_grid.set_titles(template="{value}", maxchar=40)

    return f


def plot_mean_time_height(
    th_da: xr.DataArray, vartype: str, scale: float = None
) -> plt.figure:
    th_da = th_da.assign_coords({FORECAST_TIME_DIM: th_da[FORECAST_TIME_DIM] / 60})
    if vartype == "tendencies":
        th_da.attrs.update({"units": f"{th_da.attrs['units']}/s"})
        th_da.name = f"{th_da.name} tendencies"
        if "long_name" in th_da.attrs:
            th_da.attrs.update({"long_name": f"{th_da.attrs['long_name']} tendencies"})
    facetgrid = th_da.plot(
        x=FORECAST_TIME_DIM, y="z", col="composite", yincrease=False, vmax=scale,
    )
    plt.suptitle(f"coarse model {th_da.name} across {FORECAST_TIME_DIM}")
    for ax in facetgrid.axes.flatten():
        ax.set_xlabel(f"{FORECAST_TIME_DIM} [min]")
        ax.set_xlim(
            [
                th_da[FORECAST_TIME_DIM].values[0] - 0.5,
                th_da[FORECAST_TIME_DIM].values[-1] + 0.5,
            ]
        )
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
    stride: int = None,
    scale: float = None,
) -> plt.figure:

    ds = ds.assign_coords({FORECAST_TIME_DIM: (ds[FORECAST_TIME_DIM] / 60).astype(int)})
    varnames = ["_".join([ds_name, suffix]) for suffix in composites]
    ds_across_vars = xr.Dataset(
        {
            dQ_name: xr.concat(
                [
                    ds[var].expand_dims({"dQ_composite": [suffix]})
                    for var, suffix in zip(varnames, composites)
                ],
                dim="dQ_composite",
            )
        }
    )

    if dQ_type == "tendencies":
        ds_across_vars[dQ_name] = ds_across_vars[dQ_name].assign_attrs(
            {
                "long_name": dQ_name + " tendency",
                "units": ds[varnames[0]].attrs["units"] + "/s",
            }
        )

    def _facet_line_plot(arr: np.ndarray):
        ax = plt.gca()
        ax.set_prop_cycle(
            color=["b", [1, 0.5, 0], "b", [1, 0.5, 0]], linestyle=["-", "-", "--", "--"]
        )
        nz = arr.shape[1] + 1.0
        h = ax.plot(arr.T, np.arange(1.0, nz))
        ax.plot([0, 0], [1, nz], "k--")
        return h

    facetgrid = xr.plot.FacetGrid(
        data=ds_across_vars.isel({FORECAST_TIME_DIM: slice(start, end, stride)}),
        col=FORECAST_TIME_DIM,
        col_wrap=3,
    )

    facetgrid = facetgrid.map(_facet_line_plot, dQ_name)
    facetgrid.axes.flatten()[0].invert_yaxis()
    facetgrid.axes.flatten()[0].set_ylim([ds.sizes["z"], 1])
    if scale is not None:
        facetgrid.axes.flatten()[0].set_xlim([-scale, scale])
    legend_ax = facetgrid.axes.flatten()[-2]
    handles = legend_ax.get_lines()
    legend_ax.legend(handles[:-1], composites, loc=2)
    facetgrid.set_titles(template="{value} min")
    for ax in facetgrid.axes[-1, :]:
        ax.set_xlabel(f"{dQ_name} [{ds_across_vars[dQ_name].attrs['units']}]")
        pos = ax.get_position().bounds
        pos_new = [pos[0], pos[1], pos[2], 0.75]
        ax.set_position(pos_new)
    for ax in facetgrid.axes[:, 0]:
        ax.set_ylabel("model level")
    n_rows = facetgrid.axes.shape[0]
    f = facetgrid.fig
    f.set_size_inches([12, n_rows * 4])
    f.tight_layout()
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{dQ_name}")

    return facetgrid.fig


def plot_diurnal_cycles(
    ds: xr.Dataset,
    var: str,
    start: int = None,
    end: int = None,
    stride: int = None,
    scale: float = None,
) -> plt.figure:

    ds = ds.assign_coords({FORECAST_TIME_DIM: (ds[FORECAST_TIME_DIM] / 60).astype(int)})

    def _facet_line_plot(arr: np.ndarray):
        ax = plt.gca()
        ax.plot([0.0, 24.0], [0, 0], "k-")
        ax.set_prop_cycle(color=["b", "g", [1, 0.5, 0]])
        h = ax.plot(np.arange(0.5, 24.5), arr.T)
        return h

    facetgrid = xr.plot.FacetGrid(
        data=(
            ds.isel({FORECAST_TIME_DIM: slice(start, end, stride)}).sel(
                {DELTA_DIM: ["hi-res", "coarse", "hi-res - coarse"]}
            )
        ),
        col=FORECAST_TIME_DIM,
    )

    facetgrid = facetgrid.map(_facet_line_plot, var)
    facetgrid.axes.flatten()[0].set_xlim([0, 24])
    legend_ax = facetgrid.axes.flatten()[-2]
    handles = legend_ax.get_lines()[1:]
    if var.startswith("net"):
        residual_label = "residual of tendencies"
    else:
        residual_label = "differences of physics"
    legend_ax.legend(
        handles, ["hi-res physics", "coarse physics", residual_label], loc=2
    )
    facetgrid.set_titles(template="{value} min")
    for ax in facetgrid.axes.flatten():
        ax.set_xlabel("mean local time [hrs]")
        ax.set_xticks(np.arange(0.0, 25.0, 4.0))
        ax.set_xlim([0.0, 24.0])
        pos = ax.get_position().bounds
        pos_new = [pos[0], pos[1], pos[2], 0.75]
        ax.set_position(pos_new)
    for ax in facetgrid.axes[:, 0]:
        ax.set_ylabel(f"{var} [{ds[var].attrs['units']}]")
    if scale is not None:
        ax.set_ylim([-scale, scale])
    f = facetgrid.fig
    f.set_size_inches([12, 4])
    f.tight_layout()
    f.set_dpi(FIG_DPI)
    f.suptitle(f"{var}")

    return f


def plot_timestep_counts(timesteps):
    """Create and save plots of distribution of training and test timesteps"""

    f = gallery.plot_daily_and_hourly_hist(timesteps)
    f.set_size_inches([10, 4])
    f.set_dpi(FIG_DPI)

    return f
