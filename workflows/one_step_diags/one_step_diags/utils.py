import xarray as xr
import numpy as np
import fsspec
from scipy.stats import binned_statistic
from vcm.cubedsphere.constants import TILE_COORDS
from vcm.select import mask_to_surface_type
from vcm.convenience import parse_datetime_from_str, round_time
from vcm.safe import get_variables
from vcm import thermo, local_time, net_precipitation, net_heating
import logging
from typing import Sequence, Mapping
from .config import (
    SFC_VARIABLES,
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    DELTA_DIM,
    ZARR_STEP_DIM,
    ZARR_STEP_NAMES,
)


logger = logging.getLogger("one_step_diags")

_SECONDS_PER_DAY = 86400
_MM_PER_M = 1000


def time_coord_to_datetime(
    ds: xr.Dataset, time_coord: str = INIT_TIME_DIM
) -> xr.Dataset:

    init_datetime_coords = [
        parse_datetime_from_str(timestamp) for timestamp in ds[time_coord].values
    ]
    ds = ds.assign_coords({time_coord: init_datetime_coords})

    return ds


def net_heating_from_hi_res_physics(ds: xr.Dataset) -> xr.DataArray:
    """Compute the net heating from a dataset of diagnostic output
    This should be equivalent to the vertical integral (i.e. <>) of Q1::
        cp <Q1>
    Args:
        ds: a datasets with the names for the heat fluxes and precipitation used
            by the ML pipeline
    Returns:
        the total net heating, the rate of change of the dry enthalpy <c_p T>
    """
    fluxes = (
        ds["DLWRFsfc"],
        ds["DSWRFsfc"],
        ds["ULWRFsfc"],
        ds["ULWRFtoa"],
        ds["USWRFsfc"],
        ds["USWRFtoa"],
        ds["DSWRFtoa"],
        ds["SHTFLsfc"],
        ds["PRATEsfc"],
    )
    return net_heating(*fluxes)


def load_hires_prog_diag(diag_data_path, init_times):
    """Loads coarsened diagnostic variables from the prognostic high res run.
    
    Args:
        diag_data_path (str): path to directory containing coarsened high res
            diagnostic data
        init_times (list(datetime)): list of datetimes to filter diagnostic data to
    
    Returns:
        xarray dataset: prognostic high res diagnostic variables
    """
    ds_diag = xr.open_zarr(fsspec.get_mapper(diag_data_path), consolidated=True).rename(
        {"time": INIT_TIME_DIM}
    )
    ds_diag = ds_diag.assign_coords(
        {
            INIT_TIME_DIM: [round_time(t) for t in ds_diag[INIT_TIME_DIM].values],
            "tile": TILE_COORDS,
        }
    )
    return ds_diag.sel({INIT_TIME_DIM: init_times})


def _precipitation_rate_from_total_precipitation(
    total_precipitation: xr.DataArray, time_dim: str = FORECAST_TIME_DIM
) -> xr.DataArray:
    """Compute precipitation rate from wrapper total_precipitation variable
    
    Args:
        total_precipitation: timestep precipitation in m
        time_dim: name of time dimension coordinate variable, which must provide
            output times in seconds
    
    Returns:
        precipitation_rate: precipitation rate in mm/s
    """

    dt_seconds = total_precipitation[time_dim].diff(time_dim).isel({time_dim: 0})
    precipitation_rate = _MM_PER_M * total_precipitation / dt_seconds
    precipitation_rate = precipitation_rate.assign_attrs(
        {"long name": "precipitation rate", "units": "mm/s"}
    )

    return precipitation_rate


def insert_hi_res_diags(
    ds: xr.Dataset, hi_res_diags_path: str, config: Mapping
) -> xr.Dataset:

    # temporary kluge for cumulative surface longwave from coarse diag netcdfs
    cumulative_vars = ["DLWRFsfc", "ULWRFsfc"]
    surface_longwave = (
        ds[cumulative_vars].fillna(value=0).diff(dim=FORECAST_TIME_DIM, label="upper")
    )
    ds = ds.drop(labels=cumulative_vars).merge(surface_longwave)

    # create precipitation_rate variable here so that it's equivalent to hi-res
    # PRATE
    ds = ds.assign(
        {
            "precipitation_rate": _precipitation_rate_from_total_precipitation(
                ds["total_precipitation"]
            )
        }
    )

    new_dims = config["HI_RES_DIAGS_DIMS"]
    datetimes = list(ds[INIT_TIME_DIM].values)
    ds_hires_diags = (
        load_hires_prog_diag(hi_res_diags_path, datetimes).rename(new_dims).load()
    )

    new_vars = {}
    for coarse_name, hires_name in config["HI_RES_DIAGS_MAPPING"].items():
        hires_name = hires_name + "_coarse"  # this is confusing...
        hires_var = ds_hires_diags[hires_name].transpose(
            INIT_TIME_DIM, "tile", "y", "x"
        )
        coarse_var = ds[coarse_name].load()
        coarse_var.loc[
            {FORECAST_TIME_DIM: 0, ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]}
        ] = hires_var
        new_vars[coarse_name] = coarse_var

    ds = ds.assign(new_vars)

    return ds


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
            ),
            "precipitating_water": precipitating_water_mixing_ratio,
            "cloud_water_ice": cloud_water_ice_mixing_ratio,
            "liquid_ice_temperature": thermo.liquid_ice_temperature(
                ds["air_temperature"],
                ds["cloud_ice_mixing_ratio"],
                ds["cloud_water_mixing_ratio"],
                ds["rain_mixing_ratio"],
                ds["snow_mixing_ratio"],
                ds["graupel_mixing_ratio"],
            ),
            "surface_pressure": thermo.surface_pressure_from_delp(
                ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "liquid_water_equivalent": (
                thermo.column_integrated_liquid_water_equivalent(
                    ds["specific_humidity"],
                    ds["pressure_thickness_of_atmospheric_layer"],
                )
            ),
            "column_integrated_heat": thermo.column_integrated_heat(
                ds["air_temperature"], ds["pressure_thickness_of_atmospheric_layer"]
            ),
            "net_precipitation_physics": net_precipitation(
                ds["latent_heat_flux"], ds["precipitation_rate"]
            ),
            "evaporation": thermo.surface_evaporation_mm_day_from_latent_heat_flux(
                ds["latent_heat_flux"]
            ),
            "net_heating_physics": net_heating_from_hi_res_physics(
                ds.rename(
                    {
                        "sensible_heat_flux": "SHTFLsfc",
                        "precipitation_rate": "PRATEsfc",
                    }
                )
            ),
            "precipitation_rate": (
                _SECONDS_PER_DAY * ds["precipitation_rate"]
            ).assign_attrs({"long name": "precipitation rate", "units": "mm/day"}),
        }
    )

    return ds.drop(
        labels=[
            "cloud_ice_mixing_ratio",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
        ]
        + SFC_VARIABLES
    )


def _align_time_and_concat(ds_hires: xr.Dataset, ds_coarse: xr.Dataset) -> xr.Dataset:

    ds_coarse = (
        ds_coarse.isel({INIT_TIME_DIM: [0]})
        .assign_coords({DELTA_DIM: "coarse"})
        .drop(labels=ZARR_STEP_DIM)
        .squeeze()
    )

    ds_hires = (
        ds_hires.isel({FORECAST_TIME_DIM: [0]})
        .sel({ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]})
        .drop(labels=ZARR_STEP_DIM)
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
    tendencies_coarse = ds.diff(ZARR_STEP_DIM, label="lower") / dt_forecast

    tendencies_both = _align_time_and_concat(tendencies_hires, tendencies_coarse)

    return tendencies_both


def _select_both_states_and_concat(ds: xr.Dataset) -> xr.Dataset:

    states_hires = ds.isel({INIT_TIME_DIM: slice(None, -1)})
    states_coarse = ds.sel({ZARR_STEP_DIM: ZARR_STEP_NAMES["begin"]})
    states_both = _align_time_and_concat(states_hires, states_coarse)

    return states_both


def get_states_and_tendencies(ds: xr.Dataset) -> xr.Dataset:

    tendencies = _compute_both_tendencies_and_concat(ds)
    states = _select_both_states_and_concat(ds)
    states_and_tendencies = xr.concat([states, tendencies], dim="var_type")
    states_and_tendencies = states_and_tendencies.assign_coords(
        {"var_type": ["states", "tendencies"]}
    )

    return states_and_tendencies


def insert_column_integrated_tendencies(ds: xr.Dataset) -> xr.Dataset:

    ds = ds.assign(
        {
            "column_integrated_heating": thermo.column_integrated_heating(
                ds["air_temperature"].sel({"var_type": "tendencies"}),
                ds["pressure_thickness_of_atmospheric_layer"].sel(
                    {"var_type": "states"}
                ),
            ).expand_dims({"var_type": ["tendencies"]}),
            "minus_column_integrated_moistening": (
                thermo.minus_column_integrated_moistening(
                    ds["specific_humidity"].sel({"var_type": "tendencies"}),
                    ds["pressure_thickness_of_atmospheric_layer"].sel(
                        {"var_type": "states"}
                    ),
                ).expand_dims({"var_type": ["tendencies"]})
            ),
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
                    "long_name": f"absolute {ds[var].attrs['long_name']}"
                    if "long_name" in ds[var].attrs
                    else None,
                    "units": ds[var].attrs.get("units", "none"),
                }
            )
        else:
            raise ValueError("Invalid variable name for absolute tendencies.")

    return ds


def insert_variable_at_model_level(ds: xr.Dataset, level_vars: Sequence):

    for level_var in level_vars:
        var = level_var["name"]
        level = level_var["level"]
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

    local_time = (
        local_time.stack(dimensions={"sample": stack_dims}).load().dropna("sample")
    )
    da = da.stack(dimensions={"sample": stack_dims}).load().dropna("sample")
    other_dims = [dim for dim in da.dims if dim != "sample"]
    diurnal_coords = [(dim, da[dim]) for dim in other_dims] + [
        ("local_hour_of_day", np.arange(0.0, 24.0))
    ]
    diurnal_da = xr.DataArray(coords=diurnal_coords)
    diurnal_da.name = da.name
    for valid_time in da[FORECAST_TIME_DIM]:
        da_single_time = da.sel({FORECAST_TIME_DIM: valid_time})
        try:
            bin_means, bin_edges, _ = binned_statistic(
                local_time.values, da_single_time.values, bins=np.arange(0.0, 25.0)
            )
            diurnal_da.loc[{FORECAST_TIME_DIM: valid_time}] = bin_means
        except AttributeError:
            logger.warn(
                f"Diurnal mean computation failed for initial time "
                f"{da[INIT_TIME_DIM].item()} for {da.name} "
                f"and forecast time {valid_time.item()} due to null values."
            )
            diurnal_da = None
            break

    return diurnal_da


def insert_diurnal_means(
    ds: xr.Dataset, var_mapping: Mapping, mask: str = "land_sea_mask",
) -> xr.Dataset:

    ds = ds.assign({"local_time": local_time(ds, time=INIT_TIME_DIM)})

    for domain in ["global", "land", "sea"]:

        logger.info(f"Computing diurnal means for {domain}")

        if domain in ["land", "sea"]:
            ds_domain = mask_to_surface_type(ds, domain, surface_type_var=mask)
        else:
            ds_domain = ds

        for var, attrs in var_mapping.items():

            residual_name = attrs["hi-res - coarse"]["name"]
            residual_type = attrs["hi-res - coarse"]["var_type"]
            physics_name = attrs["physics"]["name"]
            physics_type = attrs["physics"]["var_type"]

            da_residual_domain = mean_diurnal_cycle(
                ds_domain[residual_name].sel(
                    {DELTA_DIM: ["hi-res - coarse"], "var_type": residual_type}
                ),
                ds_domain["local_time"],
            )
            da_physics_domain = mean_diurnal_cycle(
                ds_domain[physics_name].sel(
                    {DELTA_DIM: ["hi-res", "coarse"], "var_type": physics_type}
                ),
                ds_domain["local_time"],
            )
            if da_residual_domain is None or da_physics_domain is None:
                return None
            else:
                ds = ds.assign(
                    {
                        f"{var}_{domain}": xr.concat(
                            [da_physics_domain, da_residual_domain], dim=DELTA_DIM
                        )
                    }
                )
                ds[f"{var}_{domain}"].attrs.update(ds[residual_name].attrs)

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
    da: xr.DataArray, weights: xr.DataArray, dims=["tile", "x", "y"]
) -> xr.DataArray:
    """Compute weighted mean of a DataArray
    Args:
        da: xr.DataArray to be averaged
        weights: xr.DataArray of weights of the dataset, must be broadcastable to ds
        dims: dimension names over which to average
    Returns
        weighted mean average da
    """

    wm = (da * weights).sum(dim=dims) / weights.sum(dim=dims)

    return wm.assign_attrs(da.attrs)


def insert_area_means(
    ds: xr.Dataset, weights: xr.DataArray, varnames: list, mask_names: list
) -> xr.Dataset:

    for var in varnames:
        if var in ds:
            new_name = f"{var}_global_mean"
            ds = ds.assign({new_name: _weighted_mean(ds[var], weights)})
        else:
            raise ValueError("Variable for global mean calculations not in dataset.")

    if "land_sea_mask" in mask_names:

        logger.info(f"Computing domain means.")

        weights_land = mask_to_surface_type(
            get_variables(ds, ["land_sea_mask"]).merge(weights),
            "land",
            surface_type_var="land_sea_mask",
        )["area"]

        weights_sea = mask_to_surface_type(
            get_variables(ds, ["land_sea_mask"]).merge(weights),
            "sea",
            surface_type_var="land_sea_mask",
        )["area"]

        for var in varnames:
            if var in ds:
                land_new_name = f"{var}_land_mean"
                sea_new_name = f"{var}_sea_mean"
                ds = ds.assign(
                    {
                        land_new_name: _weighted_mean(ds[var], weights_land),
                        sea_new_name: _weighted_mean(ds[var], weights_sea),
                    }
                )
            else:
                raise ValueError(
                    "Variable for land/sea mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names:

        logger.info(f"Computing P-E means.")

        weights_pos_PminusE = _mask_to_PminusE_sign(
            get_variables(ds, ["net_precipitation_physics"]).merge(weights),
            "positive",
            "net_precipitation_physics",
        )["area"]

        weights_neg_PminusE = _mask_to_PminusE_sign(
            get_variables(ds, ["net_precipitation_physics"]).merge(weights),
            "negative",
            "net_precipitation_physics",
        )["area"]

        for var in varnames:
            if var in ds:
                pos_new_name = f"{var}_pos_PminusE_mean"
                neg_new_name = f"{var}_neg_PminusE_mean"
                ds = ds.assign(
                    {
                        pos_new_name: _weighted_mean(ds[var], weights_pos_PminusE),
                        neg_new_name: _weighted_mean(ds[var], weights_neg_PminusE),
                    }
                )
            else:
                raise ValueError(
                    "Variable for sign(P - E) mean calculations not in dataset."
                )

    if "net_precipitation_physics" in mask_names and "land_sea_mask" in mask_names:

        logger.info(f"Computing domain + P-E means.")

        for domain in ["land", "sea"]:
            weights_pos_PminusE_domain = mask_to_surface_type(
                get_variables(ds, ["land_sea_mask"]).merge(weights_pos_PminusE),
                domain,
                surface_type_var="land_sea_mask",
            )["area"]
            weights_neg_PminusE_domain = mask_to_surface_type(
                get_variables(ds, ["land_sea_mask"]).merge(weights_neg_PminusE),
                domain,
                surface_type_var="land_sea_mask",
            )["area"]

            for var in varnames:
                if "z" in ds[var].dims:  # only do these composites for 3D vars

                    pos_domain_new_name = f"{var}_pos_PminusE_{domain}_mean"
                    neg_domain_new_name = f"{var}_neg_PminusE_{domain}_mean"
                    ds = ds.assign(
                        {
                            pos_domain_new_name: _weighted_mean(
                                ds[var], weights_pos_PminusE_domain
                            ),
                            neg_domain_new_name: _weighted_mean(
                                ds[var], weights_neg_PminusE_domain
                            ),
                        }
                    )

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


def shrink_ds(ds: xr.Dataset, config: Mapping):
    """SHrink the datast to the variables actually used in plotting
    """

    keepvars = _keepvars(config)
    dropvars = set(ds.data_vars).difference(keepvars)

    return ds.drop_vars(dropvars)


def _keepvars(config: Mapping) -> set:
    """Determine final variables in netcdf based on config
    """

    keepvars = set(
        [f"{var}_global_mean" for var in list(config["GLOBAL_MEAN_2D_VARS"])]
        + [
            f"{var}_{composite}_mean"
            for var in list(config["GLOBAL_MEAN_3D_VARS"])
            for composite in ["global", "sea", "land"]
        ]
        + [
            f"{var}_{domain}"
            for var in config["DIURNAL_VAR_MAPPING"]
            for domain in ["land", "sea", "global"]
        ]
        + [
            item
            for spec in config["DQ_MAPPING"].values()
            for item in [f"{spec['physics_name']}_physics", spec["tendency_diff_name"]]
        ]
        + [
            f"{dq_var}_{composite}"
            for dq_var in list(config["DQ_PROFILE_MAPPING"])
            for composite in list(config["PROFILE_COMPOSITES"])
        ]
        + list(config["GLOBAL_2D_MAPS"])
        + config["GRID_VARS"]
    )

    return keepvars
