import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import binned_statistic_2d
import warnings
import xarray as xr

from vcm.cubedsphere.constants import (
    INIT_TIME_DIM,
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_Z_CENTER,
    VAR_LAT_CENTER,
    PRESSURE_GRID,
)
import vcm
from vcm.cubedsphere.regridz import regrid_to_common_pressure
from vcm.select import mask_to_surface_type
from vcm.visualize import plot_cube, mappable_var

from vcm.visualize.plot_diagnostics import plot_diurnal_cycle
from fv3net.diagnostics import (
    get_latlon_grid_coords_set,
    EXAMPLE_CLIMATE_LATLON_COORDS,
)
from . import (
    integrate_for_Q,
    lower_tropospheric_stability,
    DATASET_NAME_PREDICTION, 
    DATASET_NAME_FV3_TARGET,
    DATASET_NAME_SHIELD_HIRES,
)

kg_m2s_to_mm_day = (1e3 * 86400) / 997.0
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
STACK_DIMS = ["tile", INIT_TIME_DIM, COORD_X_CENTER, COORD_Y_CENTER]
DIAG_VARS = [
    "net_precipitation",
    "net_heating", 
    "dQ1", 
    "dQ2", 
    "net_precipitation_ml", 
    "net_heating_ml", 
    "net_precipitation_physics", 
    "net_heating_physics"]
DPI_FIGURES = {
    "LTS": 100,
    "dQ2_pressure_profiles": 100,
    "R2_pressure_profiles": 100,
    "diurnal_cycle": 90,
    "map_plot_3col": 120,
}
matplotlib.use("Agg")


def make_all_plots(ds, output_dir):
    """ Makes figures for predictions on test data

    Args:
        ds: xarray dataset containing variables for diagnostic plots as well
        as a "dataset" coord that distinguishes ML prediction, FV3 target, and SHiELD 
        coarsened high res
        output_dir: location to write figures to

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """
    report_sections = {}

    # for convenience, separate the land/sea data
    ds_land = mask_to_surface_type(ds, "land")
    ds_sea = mask_to_surface_type(ds, "sea")    
    
    figs = _map_plot_ml_frac_of_total(ds)
    fig_pe_ml, fig_pe_ml_frac, fig_heating_ml, fig_heating_ml_frac = figs
    fig_pe_ml.savefig(os.path.join(output_dir, "dQ2_vertical_integral_map.png"))
    fig_pe_ml_frac.savefig(os.path.join(output_dir, "dQ2_frac_of_PE.png"))
    fig_heating_ml.savefig(os.path.join(output_dir, "dQ1_vertical_integral_map.png"))
    fig_heating_ml_frac.savefig(os.path.join(output_dir, "dQ1_frac_of_heating.png"))
    
    report_sections["ML model contributions to Q1 and Q2"] = [
        "dQ2_vertical_integral_map.png",
        "dQ2_frac_of_PE.png",
        "dQ1_vertical_integral_map.png",
        "dQ1_frac_of_heating.png",
    ]
    
    # LTS
    _plot_lower_troposphere_stability(ds, lat_max=20) \
        .savefig(os.path.join(output_dir, "LTS_vs_Q.png"), dpi=DPI_FIGURES["LTS"])
    report_sections["Lower tropospheric stability vs humidity"] = ["LTS_vs_Q.png"]
    
    # Vertical dQ2 profiles over land and ocean
    _make_vertical_profile_plots(
        ds_land, "dQ2", "[kg/kg/s]", "land: dQ2 vertical profile"
    ).savefig(
        os.path.join(output_dir, "vertical_profile_dQ2_land.png"),
        dpi=DPI_FIGURES["dQ2_pressure_profiles"],
    )
    _make_vertical_profile_plots(
        ds_sea, "dQ2", "[kg/kg/s]", "ocean: dQ2 vertical profile"
    ).savefig(
        os.path.join(output_dir, "vertical_profile_dQ2_sea.png"),
        dpi=DPI_FIGURES["dQ2_pressure_profiles"],
    )
    report_sections["dQ2 pressure level profiles"] = [
        "vertical_profile_dQ2_land.png",
        "vertical_profile_dQ2_sea.png",
    ]

    # plot P-E across the diurnal cycle
    grid = ds[["lat", "lon"]]
    local_coords = get_latlon_grid_coords_set(grid, EXAMPLE_CLIMATE_LATLON_COORDS)
    ds["local_time"] = vcm.local_time(ds)

    plot_diurnal_cycle(
        mask_to_surface_type(ds[["net_precipitation", "slmsk", "local_time"]], "sea"), "net_precipitation", title="ocean"
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_P-E_sea.png"),
        dpi=DPI_FIGURES["diurnal_cycle"],
    )
    plot_diurnal_cycle(
        mask_to_surface_type(ds[["net_precipitation", "slmsk", "local_time"]], "land"), "net_precipitation", title="land"
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_P-E_land.png"),
        dpi=DPI_FIGURES["diurnal_cycle"],
    )
    for location_name, coords in local_coords.items():
        plot_diurnal_cycle(
            ds[["net_precipitation", "local_time"]].sel(coords),
            "net_precipitation",
            title=location_name,
            ylabel="P-E [mm/day]",
        ).savefig(
            os.path.join(output_dir, f"diurnal_cycle_P-E_{location_name}.png"),
            dpi=DPI_FIGURES["diurnal_cycle"],
        )
    report_sections["Diurnal cycle, P-E"] = [
        "diurnal_cycle_P-E_sea.png",
        "diurnal_cycle_P-E_land.png",
    ] + [f"diurnal_cycle_P-E_{location_name}.png" for location_name in local_coords]

    # plot column heating across the diurnal cycle
    plot_diurnal_cycle(
        mask_to_surface_type(ds[["net_heating", "slmsk", "local_time"]], "sea"), "net_heating", title="sea"
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_heating_sea.png"),
        dpi=DPI_FIGURES["diurnal_cycle"],
    )
    plot_diurnal_cycle(
        mask_to_surface_type(ds[["net_heating", "slmsk", "local_time"]], "land"), "net_heating", title="land"
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_heating_land.png"),
        dpi=DPI_FIGURES["diurnal_cycle"],
    )

    for location_name, coords in local_coords.items():
        plot_diurnal_cycle(
            ds[["net_heating", "local_time"]].sel(coords),
            "net_heating",
            title=location_name,
            ylabel="heating [W/m$^2$]",
        ).savefig(
            os.path.join(output_dir, f"diurnal_cycle_heating_{location_name}.png"),
            dpi=DPI_FIGURES["diurnal_cycle"],
        )
    report_sections["Diurnal cycle, heating"] = [
        "diurnal_cycle_heating_sea.png",
        "diurnal_cycle_heating_land.png",
    ] + [f"diurnal_cycle_heating_{location_name}.png" for location_name in local_coords]
    
    # map plot variables and compare across prediction/ C48 /coarsened high res data
    _plot_comparison_maps(
        ds,
        "net_precipitation",
        time_index_selection=None,
        plot_cube_kwargs={"cbar_label": "time avg, P-E [mm/day]"},
    ).savefig(
        os.path.join(output_dir, "P-E_time_avg.png"), dpi=DPI_FIGURES["map_plot_3col"]
    )
    _plot_comparison_maps(
        ds,
        "net_precipitation",
        time_index_selection=[0, 2],
        plot_cube_kwargs={"cbar_label": "timestep snapshot, P-E [mm/day]"},
    ).savefig(
        os.path.join(output_dir, "P-E_time_snapshots.png"),
        dpi=DPI_FIGURES["map_plot_3col"],
    )
    report_sections["P-E"] = ["P-E_time_avg.png", "P-E_time_snapshots.png"]

    _plot_comparison_maps(
        ds,
        "net_heating",
        time_index_selection=None,
        plot_cube_kwargs={"cbar_label": "time avg, column heating [W/m$^2$]"},
    ).savefig(
        os.path.join(output_dir, "column_heating_time_avg.png"),
        dpi=DPI_FIGURES["map_plot_3col"],
    )
    _plot_comparison_maps(
        ds,
        "net_heating",
        time_index_selection=[0, -1],
        plot_cube_kwargs={"cbar_label": "timestep snapshot, column heating [W/m$^2$]"},
    ).savefig(
        os.path.join(output_dir, "column_heating_snapshots.png"),
        dpi=DPI_FIGURES["map_plot_3col"],
    )
    report_sections["Column heating"] = [
        "column_heating_time_avg.png",
        "column_heating_snapshots.png",
    ]

    return report_sections


# Below are plotting functions specific to this diagnostic workflow


def _plot_comparison_maps(
    ds, var, time_index_selection=None, plot_cube_kwargs=None
):
    # map plot a variable and compare across prediction/ C48 /coarsened high res data
    matplotlib.rcParams["figure.dpi"] = 200
    plt.clf()
    plot_cube_kwargs = plot_cube_kwargs or {}

    if not time_index_selection:
        map_var = mappable_var(ds.mean(INIT_TIME_DIM), var)
    else:
        map_var = mappable_var(
            ds.isel({INIT_TIME_DIM: time_index_selection}), var
        )
        plot_cube_kwargs["row"] = INIT_TIME_DIM
    fig = plot_cube(map_var, col="dataset", **plot_cube_kwargs)[0]
    if isinstance(time_index_selection, int):
        time_label = (
            ds[INIT_TIME_DIM]
            .values[time_index_selection]
            .strftime("%Y-%m-%d, %H:%M:%S")
        )
        plt.suptitle(time_label)
    return fig


def _make_vertical_profile_plots(ds, var, units, title=None):
    """Creates vertical profile plots of dQ2 for drying/moistening columns

    Args:
        ds (xr dataset): [description]
        var (str): [description]
        units (str): [description]
        output_dir (str): [description]
        plot_filename (str, optional): [description].
             Defaults to f"vertical_profile.png".
        title (str, optional): [description]. Defaults to None.
    """

    plt.clf()
    fig = plt.figure()
    pos_mask, neg_mask = (
        ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)["net_precipitation"] > 0,
        ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)["net_precipitation"] < 0,
    )
    ds_pred = regrid_to_common_pressure(
        ds.sel(dataset=DATASET_NAME_PREDICTION)[var], ds.sel(dataset=DATASET_NAME_PREDICTION)["delp"])
    ds_target = regrid_to_common_pressure(
        ds.sel(dataset=DATASET_NAME_FV3_TARGET)[var], ds.sel(dataset=DATASET_NAME_FV3_TARGET)["delp"])

    ds_pred_pos_PE = ds_pred.where(pos_mask)
    ds_pred_neg_PE = ds_pred.where(neg_mask)
    ds_target_pos_PE = ds_target.where(pos_mask)
    ds_target_neg_PE = ds_target.where(neg_mask)

    pressure = ds_pred.pressure.values / 100.0
    profiles_kwargs = zip(
        [ds_pred_pos_PE, ds_target_pos_PE, ds_pred_neg_PE, ds_target_neg_PE],
        [
            {"label": "P-E > 0, prediction", "color": "blue", "linestyle": "-"},
            {"label": "P-E > 0, target", "color": "blue", "linestyle": "--"},
            {"label": "P-E < 0, prediction", "color": "orange", "linestyle": "-"},
            {"label": "P-E < 0, target", "color": "orange", "linestyle": "--"},
        ],
    )

    for data, kwargs in profiles_kwargs:
        stack_dims = [dim for dim in STACK_DIMS if dim in data.dims]
        data_mean = np.mean(np.nan_to_num(data.stack(sample=stack_dims).values), axis=1)
        plt.plot(pressure, data_mean, **kwargs)

    plt.xlabel("Pressure [HPa]")
    plt.ylabel(units)
    if title:
        plt.title(title)
    plt.legend()
    return fig


def _plot_lower_troposphere_stability(ds, lat_max=20):
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    lat_mask = abs(ds[VAR_LAT_CENTER]) < lat_max

    ds_test = ds.sel(dataset=DATASET_NAME_FV3_TARGET)
    ds_test["net_precip_pred"] = ds.sel(dataset=DATASET_NAME_PREDICTION)["net_precipitation"]
    ds_test["net_precip_hires"] = ds.sel(dataset=DATASET_NAME_SHIELD_HIRES)["net_precipitation"]
    ds_test = vcm.mask_to_surface_type(ds_test, "sea") \
        .where(lat_mask).stack(sample=STACK_DIMS).dropna("sample")
    ds_test["pressure"] = vcm.pressure_at_midpoint_log(ds_test["delp"])

    Q = [
        integrate_for_Q(p, qt)
        for p, qt in zip(ds_test["pressure"].values.T, ds_test["sphum"].values.T)
    ]
    LTS = lower_tropospheric_stability(ds_test)
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(131)
    hist = ax1.hist2d(LTS.values, Q, bins=20)
    cbar1 = fig.colorbar(hist[3], ax=ax1)
    cbar1.set_label("count")
    ax1.set_xlabel("LTS [K]")
    ax1.set_ylabel("Q [mm]")

    ax2 = fig.add_subplot(132)
    bin_values_pred, x_edge, y_edge, _ = binned_statistic_2d(
        LTS.values, 
        Q, 
        ds_test["net_precip_pred"].values, 
        statistic="mean",
        bins=20
    )
    X, Y = np.meshgrid(x_edge, y_edge)
    PE = ax2.pcolormesh(X, Y, bin_values_pred.T, vmin=-10, vmax=100)
    cbar2 = fig.colorbar(PE, ax=ax2)
    cbar2.set_label("P-E [mm/d]")
    ax2.set_xlabel("LTS [K]")
    ax2.set_ylabel("Q [mm]")
    ax2.set_title("Avg predicted P-E")

    ax3 = fig.add_subplot(133)
    bin_values_hires, x_edge, y_edge, _ = binned_statistic_2d(
        LTS.values, 
        Q, 
        ds_test["net_precip_hires"].values, 
        statistic="mean", 
        bins=20
    )
    bin_error = bin_values_pred - bin_values_hires
    PE_err = ax3.pcolormesh(X, Y, bin_error.T)
    cbar3 = fig.colorbar(PE_err, ax=ax3)
    cbar3.set_label("P-E [mm/d]")
    ax3.set_xlabel("LTS [K]")
    ax3.set_ylabel("Q [mm]")
    ax3.set_title("Avg P-E error (predicted - high res)")
    return fig


def _map_plot_dQ_versus_total(ds):
    """ Produces plots of the residual components dQ of column heating
    and moistening for comparison to total quantities
    
    Args:
        ds (xarray dataset): dataset with "dataset" dimension denoting whether
        the dQ quantity was from the high-low res tendency or the ML model prediction
    
    Returns:
        Figure objects for plots of ML predictions of {column heating, P-E}
        for both the absolute ML prediction value as well as the ML
        prediction as a fraction of the total quantity (ML + physics)
    """
    ds = ds.assign(
        {
            "net_precipitation_ml_frac_of_total": ds["net_precipitation_ml"]
            / ds["net_precipitation"],
            "net_heating_ml_frac_of_total": ds["net_heating_ml"] / ds["net_heating"],
        }
    )
    fig_pe_ml = plot_cube(
        mappable_var(ds, "net_precipitation_ml").mean(INIT_TIME_DIM), col="dataset"
    )[0]
    fig_pe_ml.suptitle("P-E [mm/d]: residual dQ2")
    fig_pe_ml_frac = plot_cube(
        mappable_var(ds, "net_precipitation_ml_frac_of_total").mean(INIT_TIME_DIM),
        col="dataset",
    )[0]
    fig_pe_ml_frac.suptitle("P-E: dQ residual as fraction of total")

    fig_heating_ml = plot_cube(
        mappable_var(ds, "net_heating_ml").mean(INIT_TIME_DIM), col="dataset"
    )[0]
    fig_heating_ml.suptitle("heating [W/m$^2$], ML contribution")
    fig_heating_ml_frac = plot_cube(
        mappable_var(ds, "net_heating_ml_frac_of_total").mean(INIT_TIME_DIM),
        col="dataset",
    )[0]
    fig_heating_ml_frac.suptitle("heating: ML prediction as fraction of total")

    return fig_pe_ml, fig_pe_ml_frac, fig_heating_ml, fig_heating_ml_frac
