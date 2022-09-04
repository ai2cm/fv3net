from typing import Tuple, Sequence, MutableMapping, Hashable, Any, Mapping
from datetime import timedelta
import numpy as np
import xarray as xr
import cftime
from vcm.calc.thermo.vertically_dependent import (
    pressure_at_interface,
    pressure_at_midpoint_log,
)
from vcm.calc.thermo.constants import _SPECIFIC_HEAT_CONST_PRESSURE, _RDGAS

P_REF = 1.0e5
MINUTES_PER_HOUR = 60.0
SECONDS_PER_MINUTE = 60.0
TRACER_NAME_MAPPING: Mapping[str, str] = {  # this is specific to GFS physics
    "cloud_water_mixing_ratio": "ntcw",
    "rain_mixing_ratio": "ntrw",
    "cloud_ice_mixing_ratio": "ntiw",
    "snow_mixing_ratio": "ntsw",
    "graupel_mixing_ratio": "ntgl",
    "ozone_mixing_ratio": "ntoz",
    "cloud_amount": "ntclamt",
}

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


def model(
    rad_config: MutableMapping[Hashable, Any],
    tracer_inds: Mapping[str, int],
    time: cftime.DatetimeJulian,
    dt_atmos: float,
    nz: int,
    rank: int,
) -> MutableMapping[Hashable, Any]:
    model: MutableMapping[Hashable, Any] = {
        "me": rank,
        "levs": nz,
        "levr": nz,
        "nfxr": rad_config["nfxr"],
        "ncld": rad_config["ncld"],
        "ncnd": rad_config["ncnd"],
        "fhswr": rad_config["fhswr"],
        "fhlwr": rad_config["fhlwr"],
        # todo: why does solar hour need to be one timestep behind time to validate?
        "solhr": _solar_hour(time - timedelta(seconds=dt_atmos)),
        "lsswr": rad_config["lsswr"],
        "lslwr": rad_config["lslwr"],
        "imp_physics": rad_config["imp_physics"],
        "lgfdlmprad": rad_config["lgfdlmprad"],
        "uni_cld": rad_config["uni_cld"],
        "effr_in": rad_config["effr_in"],
        "indcld": rad_config["indcld"],
        "num_p3d": rad_config["num_p3d"],
        "npdf3d": rad_config["npdf3d"],
        "ncnvcld3d": rad_config["ncnvcld3d"],
        "lmfdeep2": rad_config["lmfdeep2"],
        "lmfshal": rad_config["lmfshal"],
        "sup": rad_config["sup"],
        "kdt": rad_config["kdt"],
        "do_sfcperts": rad_config["do_sfcperts"],
        "pertalb": rad_config["pertalb"],
        "do_only_clearsky_rad": rad_config["do_only_clearsky_rad"],
        "swhtr": rad_config["swhtr"],
        "solcon": rad_config["solcon"],
        "lprnt": rad_config["lprnt"],
        "lwhtr": rad_config["lwhtr"],
        "lssav": rad_config["lssav"],
    }
    for tracer_name, index in tracer_inds.items():
        if tracer_name in TRACER_NAME_MAPPING:
            model[TRACER_NAME_MAPPING[tracer_name]] = index
    model["ntrac"] = max(tracer_inds.values())
    return model


def _solar_hour(time: cftime.DatetimeJulian) -> float:
    return (
        time.hour
        + time.minute / MINUTES_PER_HOUR
        + time.second / (MINUTES_PER_HOUR * SECONDS_PER_MINUTE)
    )


def statein(
    state: State, tracer_inds: Mapping[str, int], ivflip: int, unstacked_dim="z",
) -> Mapping[str, np.ndarray]:
    state_names = [
        "pressure_thickness_of_atmospheric_layer",
        "air_temperature",
    ]
    tracer_names = list(tracer_inds.keys())
    state_names.extend(tracer_names)
    states = xr.Dataset({name: state[name] for name in state_names})
    stacked = _stack(states, unstacked_dim)
    delp = stacked["pressure_thickness_of_atmospheric_layer"]
    pi = pressure_at_interface(delp, dim_center=unstacked_dim, dim_outer="z_interface")
    pm = pressure_at_midpoint_log(delp, dim=unstacked_dim)
    exn = (pm / P_REF) ** (_RDGAS / _SPECIFIC_HEAT_CONST_PRESSURE)  # exner pressure
    temperature = stacked["air_temperature"]

    tracer_array = np.zeros(
        (stacked.sizes["column"], stacked.sizes[unstacked_dim], len(tracer_names))
    )
    for tracer_name, index in tracer_inds.items():
        tracer_array[:, :, index - 1] = stacked[tracer_name].values

    _statein = {
        "prsi": pi.values,
        "prsl": pm.values,
        "tgrs": temperature.values,
        "prslk": exn.values,
        "qgrs": tracer_array,
    }
    if ivflip == 1:
        _statein = {name: np.flip(_statein[name], axis=1) for name in _statein}
    return _statein


def _stack(
    ds: xr.Dataset, unstacked_dim: str = "z", sample_dim: str = "column"
) -> xr.Dataset:
    stack_dims = [dim for dim in ds.dims if dim != unstacked_dim]
    stacked = ds.stack({sample_dim: stack_dims})
    if unstacked_dim in ds.dims:
        stacked = stacked.transpose(sample_dim, unstacked_dim)
    return stacked


def unstack(
    data: Mapping[str, np.ndarray],
    coords: xr.DataArray,
    sample_dim: str = "column",
    unstacked_dim: str = "z",
) -> Diagnostics:
    out: Diagnostics = {}
    for name, arr in data.items():
        if arr.ndim == 1:
            da = xr.DataArray(arr, dims=[sample_dim], coords={sample_dim: coords})
        elif arr.ndim == 2:
            da = xr.DataArray(
                arr, dims=[sample_dim, unstacked_dim], coords={sample_dim: coords}
            )
        out[name] = da.unstack(dim=sample_dim)
    return out


def grid(state: State) -> Tuple[Mapping[str, np.ndarray], xr.DataArray]:
    grid_names = ["longitude", "latitude"]
    stacked_grid = _stack(
        xr.Dataset({name: state[name] for name in grid_names}), sample_dim="column"
    )
    coords = stacked_grid.column
    lon_rad = stacked_grid["longitude"].values
    lat_rad = stacked_grid["latitude"].values
    sinlat = np.sin(lat_rad)
    coslat = np.cos(lat_rad)
    return (
        {"xlon": lon_rad, "xlat": lat_rad, "sinlat": sinlat, "coslat": coslat},
        coords,
    )


SFC_VAR_MAPPING = {
    "tsfc": "surface_temperature",
    "slmsk": "land_sea_mask",
    "snowd": "snow_depth_water_equivalent",
    "sncovr": "snow_cover_in_fraction",
    "snoalb": "maximum_snow_albedo_in_fraction",
    "zorl": "surface_roughness",
    "hprime": "orographic_variables",
    "alvsf": "mean_visible_albedo_with_strong_cosz_dependency",
    "alnsf": "mean_near_infrared_albedo_with_strong_cosz_dependency",
    "alvwf": "mean_visible_albedo_with_weak_cosz_dependency",
    "alnwf": "mean_near_infrared_albedo_with_weak_cosz_dependency",
    "facsf": "fractional_coverage_with_strong_cosz_dependency",
    "facwf": "fractional_coverage_with_weak_cosz_dependency",
    "fice": "ice_fraction_over_open_water",
    "tisfc": "surface_temperature_over_ice_fraction",
}


def sfcprop(
    state: State, sfc_var_mapping: Mapping[str, str] = SFC_VAR_MAPPING
) -> Mapping[str, np.ndarray]:
    sfc = xr.Dataset({k: state[v] for k, v in list(sfc_var_mapping.items())})
    # we only want the first of "hprime" variables
    sfc["hprime"] = sfc["hprime"].isel({"orographic_variable": 0})
    stacked_sfc = _stack(sfc)
    return {name: stacked_sfc[name].values for name in stacked_sfc.data_vars}


OUT_NAMES = [
    "clear_sky_downward_longwave_flux_at_surface_python",
    "clear_sky_downward_shortwave_flux_at_surface_python",
    "clear_sky_upward_longwave_flux_at_surface_python",
    "clear_sky_upward_shortwave_flux_at_surface_python",
    "clear_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "clear_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
    "total_sky_downward_longwave_flux_at_surface_python",
    "total_sky_downward_shortwave_flux_at_surface_python",
    "total_sky_upward_longwave_flux_at_surface_python",
    "total_sky_upward_shortwave_flux_at_surface_python",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere_python",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
]


def rename_out(
    out: Tuple[Mapping[str, np.ndarray], ...], out_names: Sequence[str] = OUT_NAMES
) -> Mapping[str, np.ndarray]:
    radtendout, diagout, _ = out
    renamed = {}
    for name in out_names:
        obj = radtendout if "surface" in name else diagout
        level = "sfc" if "surface" in name else "top"
        band = "lw" if "longwave" in name else "sw"
        direction = "up" if "upward" in name else "dn"
        cloud_type = "0" if "clear" in name else "c"
        renamed[name] = obj[f"{level}f{band}"][f"{direction}fx{cloud_type}"]
    return renamed
