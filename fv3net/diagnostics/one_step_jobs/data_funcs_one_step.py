import xarray as xr
import numpy as np
from scipy.stats import binned_statistic
from vcm.cubedsphere.constants import TIME_FMT
from vcm.constants import kg_m2s_to_mm_day
from vcm.select import mask_to_surface_type
from vcm.convenience import parse_datetime_from_str
from vcm import local_time
from datetime import datetime, timedelta
import logging
from typing import Sequence, Tuple, Mapping

from fv3net.diagnostics.one_step_jobs import (
    SFC_VARIABLES,
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    DELTA_DIM,
    VAR_TYPE_DIM,
    STEP_DIM,
    GLOBAL_MEAN_2D_VARS,
    GLOBAL_MEAN_3D_VARS,
    DIURNAL_VAR_MAPPING,
    DQ_MAPPING,
    DQ_PROFILE_MAPPING,
    PROFILE_COMPOSITES,
    GLOBAL_2D_MAPS,
    GRID_VARS,
    thermo,
)
from fv3net.pipelines.common import subsample_timesteps_at_interval
from fv3net.pipelines.create_training_data.helpers import load_hires_prog_diag
from fv3net.diagnostics.data import net_heating_from_dataset

from vcm import net_precipitation

logger = logging.getLogger("one_step_diags")


def time_inds_to_open(time_coord: xr.DataArray, n_sample: int) -> Sequence:

    timestamp_list = list(time_coord[INIT_TIME_DIM].values)
    timestamp_list_no_last = timestamp_list[:-1]
    duration_minutes, restart_interval_minutes = _duration_and_interval(
        timestamp_list_no_last
    )
    sampling_interval_minutes = _sampling_interval_minutes(
        duration_minutes, restart_interval_minutes, n_sample
    )
    timestamp_subset = subsample_timesteps_at_interval(
        timestamp_list_no_last, sampling_interval_minutes
    )
    timestamp_subset_indices = [
        _timestamp_pairs_indices(timestamp_list, timestamp, restart_interval_minutes)
        for timestamp in timestamp_subset
    ]
    timestep_subset_indices = filter(
        lambda x: True if x[0] is not None else False, timestamp_subset_indices
    )

    return timestep_subset_indices


def _duration_and_interval(timestamp_list: list) -> (int, int):

    first_and_last = [datetime.strptime(timestamp_list[i], TIME_FMT) for i in (0, -1)]
    first_and_second = [datetime.strptime(timestamp_list[i], TIME_FMT) for i in (0, 1)]
    duration_minutes = (first_and_last[1] - first_and_last[0]).seconds / 60 + (
        first_and_last[1] - first_and_last[0]
    ).days * 1440
    n_steps = len(timestamp_list)
    restart_interval_minutes = duration_minutes // (n_steps - 1)
    restart_interval_minutes_first = (
        first_and_second[1] - first_and_second[0]
    ).seconds / 60
    if restart_interval_minutes != restart_interval_minutes_first:
        logger.info("Incomplete initialization set detected in .zarr")
        restart_interval_minutes = restart_interval_minutes_first

    return duration_minutes, restart_interval_minutes


def _sampling_interval_minutes(
    duration_minutes: int, restart_interval_minutes: int, n_sample: int
) -> int:
    try:
        interval_minutes = restart_interval_minutes * max(
            (duration_minutes / (n_sample - 1)) // restart_interval_minutes, 1
        )
    except ZeroDivisionError:
        interval_minutes = None
    return interval_minutes


def _timestamp_pairs_indices(
    timestamp_list: list,
    timestamp: str,
    time_fmt: str = TIME_FMT,
    time_interval_minutes: int = 15,
) -> Tuple:

    timestamp1 = datetime.strftime(
        datetime.strptime(timestamp, TIME_FMT)
        + timedelta(minutes=time_interval_minutes),
        TIME_FMT,
    )
    try:
        index0 = timestamp_list.index(timestamp)
        index1 = timestamp_list.index(timestamp1)
        pairs = index0, index1
    except ValueError:
        pairs = None, None
    return pairs


def time_coord_to_datetime(
    ds: xr.Dataset, time_coord: str = INIT_TIME_DIM
) -> xr.Dataset:

    init_datetime_coords = [
        parse_datetime_from_str(timestamp) for timestamp in ds[time_coord].values
    ]
    ds = ds.assign_coords({time_coord: init_datetime_coords})

    return ds


def insert_hi_res_diags(
    ds: xr.Dataset, hi_res_diags_path: str, varnames_mapping: Mapping
) -> xr.Dataset:

    new_dims = {"grid_xt": "x", "grid_yt": "y", "initialization_time": INIT_TIME_DIM}

    datetimes = list(ds[INIT_TIME_DIM].values)
    ds_hires_diags = load_hires_prog_diag(hi_res_diags_path, datetimes).rename(new_dims)

    new_vars = {}
    for coarse_name, hires_name in varnames_mapping.items():
        hires_name = hires_name + "_coarse"  # this is confusing...
        hires_var = ds_hires_diags[hires_name].transpose(
            INIT_TIME_DIM, "tile", "y", "x"
        )
        coarse_var = ds[coarse_name].load()
        coarse_var.loc[{FORECAST_TIME_DIM: 0, STEP_DIM: "begin"}] = hires_var
        new_vars[coarse_name] = coarse_var

    return ds.assign(new_vars)


def insert_derived_vars_from_ds_zarr(ds: xr.Dataset) -> xr.Dataset:
    """Add derived vars (combinations of direct output variables) to dataset"""

    cloud_water_ice_mixing_ratio = (
        ds["cloud_ice_mixing_ratio"] + ds["cloud_water_mixing_ratio"]
    )
    cloud_water_ice_mixing_ratio.attrs.update(
        {"long_name": "cloud water and ice mixing ratio", "units": "kg/kg"}
    )

    precipitating_water_mixing_ratio = (
        ds["rain_mixing_ratio"] + ds["snow_mixing_ratio"] + ds["graupel_mixing_ratio"]
    )
    precipitating_water_mixing_ratio.attrs.update(
        {"long_name": "precipitating water mixing ratio", "units": "kg/kg"}
    )

    ds = ds.assign(
        {
            "total_water": thermo.total_water(
                ds["specific_humidity"],
                ds["cloud_ice_mixing_ratio"],
                ds["cloud_water_mixing_ratio"],
                ds["rain_mixing_ratio"],
                ds["snow_mixing_ratio"],
                ds["graupel_mixing_ratio"],
                ds["pressure_thickness_of_atmospheric_layer"],
            ),
            "precipitating_water": precipitating_water_mixing_ratio,
            "cloud_water_ice": cloud_water_ice_mixing_ratio,
            "psurf": thermo.psurf_from_delp(
                ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "precipitable_water": thermo.precipitable_water(
                ds["specific_humidity"], ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "total_heat": thermo.total_heat(
                ds["air_temperature"], ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "net_precipitation_physics": net_precipitation(
                ds["latent_heat_flux"], ds["total_precipitation"]
            ),
            "net_heating_physics": net_heating_from_dataset(
                ds.rename(
                    {
                        "sensible_heat_flux": "SHTFLsfc",
                        "total_precipitation": "PRATEsfc",
                    }
                )
            ),
            "total_precipitation": (
                (ds["total_precipitation"] * kg_m2s_to_mm_day).assign_attrs(
                    {"long name": "total precipitation", "units": "mm/day"}
                )
            ),
        }
    )

    return ds.drop(
        labels=(
            "cloud_ice_mixing_ratio",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
        )
        + SFC_VARIABLES
    )


def _align_time_and_concat(ds_hires: xr.Dataset, ds_coarse: xr.Dataset) -> xr.Dataset:

    ds_coarse = (
        ds_coarse.isel({INIT_TIME_DIM: [0]})
        .assign_coords({DELTA_DIM: "coarse"})
        .drop(labels=STEP_DIM)
        .squeeze()
    )

    ds_hires = (
        ds_hires.isel({FORECAST_TIME_DIM: [0]})
        .sel({STEP_DIM: "begin"})
        .drop(labels=STEP_DIM)
        .assign_coords({DELTA_DIM: "hi-res"})
        .squeeze()
    )

    dim_excl = [dim for dim in ds_hires.dims if dim != FORECAST_TIME_DIM]
    ds_hires = ds_hires.broadcast_like(ds_coarse, exclude=dim_excl)

    return xr.concat([ds_hires, ds_coarse], dim=DELTA_DIM)


def _compute_both_tendencies_and_concat(ds: xr.Dataset) -> xr.Dataset:

    dt_init = ds[INIT_TIME_DIM].diff(INIT_TIME_DIM).isel(
        {INIT_TIME_DIM: 0}
    ) / np.timedelta64(1, "s")
    tendencies_hires = ds.diff(INIT_TIME_DIM, label="lower") / dt_init

    dt_forecast = (
        ds[FORECAST_TIME_DIM].diff(FORECAST_TIME_DIM).isel({FORECAST_TIME_DIM: 0})
    )
    tendencies_coarse = ds.diff(STEP_DIM, label="lower") / dt_forecast

    tendencies_both = _align_time_and_concat(tendencies_hires, tendencies_coarse)

    return tendencies_both


def _select_both_states_and_concat(ds: xr.Dataset) -> xr.Dataset:

    states_hires = ds.isel({INIT_TIME_DIM: slice(None, -1)})
    states_coarse = ds.sel({STEP_DIM: "begin"})
    states_both = _align_time_and_concat(states_hires, states_coarse)

    return states_both


def get_states_and_tendencies(ds: xr.Dataset) -> xr.Dataset:

    tendencies = _compute_both_tendencies_and_concat(ds)
    states = _select_both_states_and_concat(ds)
    states_and_tendencies = xr.concat([states, tendencies], dim=VAR_TYPE_DIM)
    states_and_tendencies = states_and_tendencies.assign_coords(
        {VAR_TYPE_DIM: ["states", "tendencies"]}
    )

    return states_and_tendencies


def insert_column_integrated_tendencies(ds: xr.Dataset) -> xr.Dataset:

    ds = ds.assign(
        {
            "column_integrated_heating": thermo.column_integrated_heating(
                ds["air_temperature"].sel({VAR_TYPE_DIM: "tendencies"}),
                ds["pressure_thickness_of_atmospheric_layer"].sel(
                    {VAR_TYPE_DIM: "states"}
                ),
            ).expand_dims({VAR_TYPE_DIM: ["tendencies"]}),
            "column_integrated_moistening": thermo.column_integrated_moistening(
                -ds["specific_humidity"].sel({VAR_TYPE_DIM: "tendencies"}),
                ds["pressure_thickness_of_atmospheric_layer"].sel(
                    {VAR_TYPE_DIM: "states"}
                ),
            ).expand_dims({VAR_TYPE_DIM: ["tendencies"]}),
        }
    )

    return ds


def insert_model_run_differences(ds: xr.Dataset) -> xr.Dataset:

    ds_residual = (
        ds.sel({DELTA_DIM: "hi-res"}) - ds.sel({DELTA_DIM: "coarse"})
    ).assign_coords({DELTA_DIM: "hi-res - coarse"})

    # combine into one dataset
    return xr.concat([ds, ds_residual], dim=DELTA_DIM)


def insert_abs_vars(ds: xr.Dataset, varnames: Sequence) -> xr.Dataset:

    for var in varnames:
        if var in ds:
            ds[var + "_abs"] = np.abs(ds[var])
            ds[var + "_abs"].attrs.update(
                {
                    "long_name": f"absolute {ds[var].attrs['long_name']}",
                    "units": ds[var].attrs["units"],
                }
            )
        else:
            raise ValueError("Invalid variable name for absolute tendencies.")

    return ds


def insert_variable_at_model_level(ds: xr.Dataset, varnames: list, level: int):

    for var in varnames:
        if var in ds:
            new_name = f"{var}_level_{level}"
            ds = ds.assign({new_name: ds[var].sel({"z": level})})
            ds[new_name].attrs.update({"long_name": f"{var} at model level {level}"})
        else:
            raise ValueError("Invalid variable for model level selection.")

    return ds


def mean_diurnal_cycle(
    da: xr.DataArray, local_time: xr.DataArray, stack_dims: list = ["x", "y", "tile"]
) -> xr.DataArray:

    local_time = local_time.stack(dimensions={"sample": stack_dims}).dropna("sample")
    da = da.stack(dimensions={"sample": stack_dims}).dropna("sample")
    other_dims = [dim for dim in da.dims if dim != "sample"]
    diurnal_coords = [(dim, da[dim]) for dim in other_dims] + [
        ("mean_local_time", np.arange(0.0, 24.0))
    ]
    diurnal_da = xr.DataArray(coords=diurnal_coords)
    for valid_time in da[FORECAST_TIME_DIM]:
        da_single_time = da.sel({FORECAST_TIME_DIM: valid_time})
        bin_means, _, _ = binned_statistic(
            local_time.values, da_single_time.values, bins=np.arange(0.0, 25.0)
        )
        diurnal_da.loc[{FORECAST_TIME_DIM: valid_time}] = bin_means

    return diurnal_da


def insert_diurnal_means(
    ds: xr.Dataset,
    var_mapping: Mapping = DIURNAL_VAR_MAPPING,
    mask: str = "land_sea_mask",
) -> xr.Dataset:

    ds = ds.assign({"local_time": local_time(ds, time=INIT_TIME_DIM)})

    for var, attrs in var_mapping.items():

        residual_name = attrs["hi-res - coarse"]["name"]
        residual_type = attrs["hi-res - coarse"][VAR_TYPE_DIM]
        hires_name = attrs["hi-res"]["name"]
        hires_type = attrs["hi-res"][VAR_TYPE_DIM]

        ds_land = mask_to_surface_type(
            ds[[residual_name, hires_name, "local_time", mask]],
            "land",
            surface_type_var=mask,
        )
        da_residual_land = mean_diurnal_cycle(
            ds_land[residual_name].sel(
                {DELTA_DIM: "hi-res - coarse", VAR_TYPE_DIM: residual_type}
            ),
            ds_land["local_time"],
        )
        da_hires_land = mean_diurnal_cycle(
            ds_land[hires_name].sel({DELTA_DIM: "hi-res", VAR_TYPE_DIM: hires_type}),
            ds_land["local_time"],
        )
        ds = ds.assign(
            {
                f"{var}_land": xr.concat(
                    [da_hires_land, da_residual_land], dim=DELTA_DIM
                ).assign_coords({DELTA_DIM: ["hi-res", "hi-res - coarse"]})
            }
        )
        ds[f"{var}_land"].attrs.update(ds[residual_name].attrs)

        ds_sea = mask_to_surface_type(
            ds[[residual_name, hires_name, "local_time", mask]],
            "sea",
            surface_type_var=mask,
        )
        da_residual_sea = mean_diurnal_cycle(
            ds_sea[residual_name].sel(
                {DELTA_DIM: "hi-res - coarse", VAR_TYPE_DIM: residual_type}
            ),
            ds_sea["local_time"],
        )
        da_hires_sea = mean_diurnal_cycle(
            ds_sea[hires_name].sel({DELTA_DIM: "hi-res", VAR_TYPE_DIM: hires_type}),
            ds_sea["local_time"],
        )
        ds = ds.assign(
            {
                f"{var}_sea": xr.concat(
                    [da_hires_sea, da_residual_sea], dim=DELTA_DIM
                ).assign_coords({DELTA_DIM: ["hi-res", "hi-res - coarse"]})
            }
        )
        ds[f"{var}_sea"].attrs.update(ds[residual_name].attrs)

    return ds


def _mask_to_PminusE_sign(
    ds: xr.Dataset, sign: str, PminusE_varname: str, PminusE_dataset: str = "hi-res"
) -> xr.Dataset:
    """
    Args:
        ds: xarray dataset
        sign: one of ['positive', 'negative']
        PminusE_varname: Name of the P - E var in ds.
        PminusE_dataset: Name of the P - E dataset to use, optional.
            Defaults to 'hi-res'.
    Returns:
        input dataset masked to the P - E sign specified
    """
    if sign in ["none", "None", None]:
        logger.info("surface_type provided as None: no mask applied.")
        return ds
    elif sign not in ["positive", "negative"]:
        raise ValueError("Must mask to either positive or negative.")

    PminusE = ds[PminusE_varname].sel({DELTA_DIM: "hi-res"})
    mask = PminusE > 0 if sign == "positive" else PminusE < 0

    return ds.where(mask)


def _weighted_mean(
    ds: xr.Dataset, weights: xr.DataArray, dims=["tile", "x", "y"]
) -> xr.Dataset:
    """Compute weighted mean of a dataset
    Args:
        ds: dataset to be averaged
        weights: xr.DataArray of weights of the dataset, must be broadcastable to ds
        dims: dimension names over which to average
    Returns
        weighted mean average ds
    """

    return (ds * weights).sum(dim=dims) / weights.sum(dim=dims)


def insert_area_means(
    ds: xr.Dataset, weights: xr.DataArray, varnames: list, mask_names: list
) -> xr.Dataset:

    wm = _weighted_mean(ds, weights)
    for var in varnames:
        if var in ds:
            new_name = f"{var}_global_mean"
            ds = ds.assign({new_name: wm[var]})
            ds[new_name] = ds[new_name].assign_attrs(ds[var].attrs)
        else:
            raise ValueError("Variable for global mean calculations not in dataset.")

    if "land_sea_mask" in mask_names:

        ds_land = mask_to_surface_type(
            ds.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_land = ds_land["area"]
        wm_land = _weighted_mean(ds_land, weights_land)

        ds_sea = mask_to_surface_type(
            ds.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_sea = ds_sea["area"]
        wm_sea = _weighted_mean(ds_sea, weights_sea)

        for var in varnames:
            if var in ds:
                land_new_name = f"{var}_land_mean"
                sea_new_name = f"{var}_sea_mean"
                ds = ds.assign({land_new_name: wm_land[var], sea_new_name: wm_sea[var]})
                ds[land_new_name] = ds[land_new_name].assign_attrs(ds[var].attrs)
                ds[sea_new_name] = ds[sea_new_name].assign_attrs(ds[var].attrs)
            else:
                raise ValueError(
                    "Variable for land/sea mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names:

        ds_pos_PminusE = _mask_to_PminusE_sign(
            ds, "positive", "net_precipitation_physics"
        )
        weights_pos_PminusE = ds_pos_PminusE["area"]
        wm_pos_PminusE = _weighted_mean(ds_pos_PminusE, weights_pos_PminusE)

        ds_neg_PminusE = _mask_to_PminusE_sign(
            ds, "negative", "net_precipitation_physics"
        )
        weights_neg_PminusE = ds_neg_PminusE["area"]
        wm_neg_PminusE = _weighted_mean(ds_neg_PminusE, weights_neg_PminusE)

        for var in varnames:
            if var in ds:
                pos_new_name = f"{var}_pos_PminusE_mean"
                neg_new_name = f"{var}_neg_PminusE_mean"
                ds = ds.assign(
                    {
                        pos_new_name: wm_pos_PminusE[var],
                        neg_new_name: wm_neg_PminusE[var],
                    }
                )
                ds[pos_new_name] = ds[pos_new_name].assign_attrs(ds[var].attrs)
                ds[neg_new_name] = ds[neg_new_name].assign_attrs(ds[var].attrs)
            else:
                raise ValueError(
                    "Variable for sign(P - E) mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names and "land_sea_mask" in mask_names:

        ds_pos_PminusE_land = mask_to_surface_type(
            ds_pos_PminusE.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_pos_PminusE_land = ds_pos_PminusE_land["area"]
        wm_pos_PminusE_land = _weighted_mean(
            ds_pos_PminusE_land, weights_pos_PminusE_land
        )

        ds_pos_PminusE_sea = mask_to_surface_type(
            ds_pos_PminusE.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_pos_PminusE_sea = ds_pos_PminusE_sea["area"]
        wm_pos_PminusE_sea = _weighted_mean(ds_pos_PminusE_sea, weights_pos_PminusE_sea)

        ds_neg_PminusE_land = mask_to_surface_type(
            ds_neg_PminusE.merge(weights), "land", surface_type_var="land_sea_mask"
        )
        weights_neg_PminusE_land = ds_neg_PminusE_land["area"]
        wm_neg_PminusE_land = _weighted_mean(
            ds_neg_PminusE_land, weights_neg_PminusE_land
        )

        ds_neg_PminusE_sea = mask_to_surface_type(
            ds_neg_PminusE.merge(weights), "sea", surface_type_var="land_sea_mask"
        )
        weights_neg_PminusE_sea = ds_neg_PminusE_sea["area"]
        wm_neg_PminusE_sea = _weighted_mean(ds_neg_PminusE_sea, weights_neg_PminusE_sea)

        for var in varnames:
            if "z" in ds[var].dims:
                pos_land_new_name = f"{var}_pos_PminusE_land_mean"
                neg_land_new_name = f"{var}_neg_PminusE_land_mean"
                pos_sea_new_name = f"{var}_pos_PminusE_sea_mean"
                neg_sea_new_name = f"{var}_neg_PminusE_sea_mean"
                ds = ds.assign(
                    {
                        pos_land_new_name: wm_pos_PminusE_land[var],
                        neg_land_new_name: wm_neg_PminusE_land[var],
                        pos_sea_new_name: wm_pos_PminusE_sea[var],
                        neg_sea_new_name: wm_neg_PminusE_sea[var],
                    }
                )
                ds[pos_land_new_name] = ds[pos_land_new_name].assign_attrs(
                    ds[var].attrs
                )
                ds[neg_land_new_name] = ds[neg_land_new_name].assign_attrs(
                    ds[var].attrs
                )
                ds[pos_sea_new_name] = ds[pos_sea_new_name].assign_attrs(ds[var].attrs)
                ds[neg_sea_new_name] = ds[neg_sea_new_name].assign_attrs(ds[var].attrs)

    if any(
        [
            mask not in ["land_sea_mask", "net_precipitation_physics"]
            for mask in mask_names
        ]
    ):
        raise ValueError(
            'Only "land_sea_mask" and "net_precipitation_physics" are '
            "suppored as masks."
        )

    return ds


def shrink_ds(ds: xr.Dataset):
    """SHrink the datast to the variables actually used in plotting
    """
    keepvars = set(
        [f"{var}_global_mean" for var in list(GLOBAL_MEAN_2D_VARS)]
        + [
            f"{var}_{composite}_mean"
            for var in list(GLOBAL_MEAN_3D_VARS)
            for composite in ["global", "sea", "land"]
        ]
        + [
            f"{var}_{domain}"
            for var in DIURNAL_VAR_MAPPING
            for domain in ["land", "sea"]
        ]
        + [
            item
            for spec in DQ_MAPPING.values()
            for item in [f"{spec['hi-res_name']}_physics", spec["hi-res - coarse_name"]]
        ]
        + [
            f"{dq_var}_{composite}"
            for dq_var in list(DQ_PROFILE_MAPPING)
            for composite in list(PROFILE_COMPOSITES)
        ]
        + list(GLOBAL_2D_MAPS)
        + list(GRID_VARS)
    )

    dropvars = set(ds.data_vars).difference(keepvars)

    return ds.drop_vars(dropvars)
