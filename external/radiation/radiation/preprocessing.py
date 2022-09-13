from typing import Tuple, Set, MutableMapping, Hashable, Any, Mapping, Sequence
from datetime import timedelta
import numpy as np
import xarray as xr
import cftime
from radiation.config import RadiationConfig

try:
    from vcm.calc.thermo.vertically_dependent import (
        pressure_at_interface,
        pressure_at_midpoint_log,
    )
    from vcm.calc.thermo.constants import _SPECIFIC_HEAT_CONST_PRESSURE, _RDGAS
except ImportError:
    pass

P_REF: float = 1.0e5
MINUTES_PER_HOUR: float = 60.0
SECONDS_PER_MINUTE: float = 60.0
STATE_NAMES_IN: Sequence[str] = [
    "air_temperature",
    "pressure_thickness_of_atmospheric_layer",
]
GRID_NAMES_IN: Sequence[str] = ["longitude", "latitude"]
TRACER_NAMES_IN_MAPPING: Mapping[str, str] = {  # this is specific to GFS physics
    "cloud_water_mixing_ratio": "ntcw",
    "rain_mixing_ratio": "ntrw",
    "cloud_ice_mixing_ratio": "ntiw",
    "snow_mixing_ratio": "ntsw",
    "graupel_mixing_ratio": "ntgl",
    "ozone_mixing_ratio": "ntoz",
    "cloud_amount": "ntclamt",
}
SFC_NAMES_IN_MAPPING: Mapping[str, str] = {
    "surface_temperature": "tsfc",
    "land_sea_mask": "slmsk",
    "snow_depth_water_equivalent": "snowd",
    "snow_cover_in_fraction": "sncovr",
    "maximum_snow_albedo_in_fraction": "snoalb",
    "surface_roughness": "zorl",
    "orographic_variables": "hprime",
    "mean_visible_albedo_with_strong_cosz_dependency": "alvsf",
    "mean_near_infrared_albedo_with_strong_cosz_dependency": "alnsf",
    "mean_visible_albedo_with_weak_cosz_dependency": "alvwf",
    "mean_near_infrared_albedo_with_weak_cosz_dependency": "alnwf",
    "fractional_coverage_with_strong_cosz_dependency": "facsf",
    "fractional_coverage_with_weak_cosz_dependency": "facwf",
    "ice_fraction_over_open_water": "fice",
    "surface_temperature_over_ice_fraction": "tisfc",
}
RENAME_OUT: Mapping[str, str] = {
    "sfcflw_dnfx0": "clear_sky_downward_longwave_flux_at_surface_python",
    "sfcfsw_dnfx0": "clear_sky_downward_shortwave_flux_at_surface_python",
    "sfcflw_upfx0": "clear_sky_upward_longwave_flux_at_surface_python",
    "sfcfsw_upfx0": "clear_sky_upward_shortwave_flux_at_surface_python",
    "topflw_upfx0": "clear_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "topfsw_upfx0": "clear_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
    "sfcflw_dnfxc": "total_sky_downward_longwave_flux_at_surface_python",
    "sfcfsw_dnfxc": "total_sky_downward_shortwave_flux_at_surface_python",
    "sfcflw_upfxc": "total_sky_upward_longwave_flux_at_surface_python",
    "sfcfsw_upfxc": "total_sky_upward_shortwave_flux_at_surface_python",
    "topfsw_dnfxc": "total_sky_downward_shortwave_flux_at_top_of_atmosphere_python",
    "topflw_upfxc": "total_sky_upward_longwave_flux_at_top_of_atmosphere_python",
    "topfsw_upfxc": "total_sky_upward_shortwave_flux_at_top_of_atmosphere_python",
    "htrlw": "total_sky_longwave_heating_rate_python",
    "lwhc": "clear_sky_longwave_heating_rate_python",
    "htrsw": "total_sky_shortwave_heating_rate_python",
    "swhc": "clear_sky_shortwave_heating_rate_python",
}
BASE_INPUT_VARIABLE_NAMES: Sequence[str] = (
    list(STATE_NAMES_IN) + list(GRID_NAMES_IN) + list(SFC_NAMES_IN_MAPPING.keys())
)
OUTPUT_VARIABLE_NAMES: Sequence[str] = list(RENAME_OUT.values())

State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


def get_model(
    rad_config: RadiationConfig,
    tracer_inds: Mapping[str, int],
    time: cftime.DatetimeJulian,
    dt_atmos: float,
    nz: int,
    rank: int,
    tracer_name_mapping: Mapping[str, str] = TRACER_NAMES_IN_MAPPING,
) -> MutableMapping[Hashable, Any]:
    model: MutableMapping[Hashable, Any] = {
        "me": rank,
        "levs": nz,
        "levr": nz,
        "nfxr": rad_config.nfxr,
        "ncld": rad_config.ncld,
        "ncnd": rad_config.ncnd,
        "fhswr": rad_config.fhswr,
        "fhlwr": rad_config.fhlwr,
        # todo: why does solar hour need to be one timestep behind time to validate?
        "solhr": _solar_hour(time - timedelta(seconds=dt_atmos)),
        "lsswr": rad_config.lsswr,
        "lslwr": rad_config.lslwr,
        "imp_physics": rad_config.imp_physics,
        "lgfdlmprad": rad_config.lgfdlmprad,
        "uni_cld": rad_config.uni_cld,
        "effr_in": rad_config.effr_in,
        "indcld": rad_config.indcld,
        "num_p3d": rad_config.num_p3d,
        "npdf3d": rad_config.npdf3d,
        "ncnvcld3d": rad_config.ncnvcld3d,
        "lmfdeep2": rad_config.lmfdeep2,
        "lmfshal": rad_config.lmfshal,
        "sup": rad_config.sup,
        "kdt": rad_config.kdt,
        "do_sfcperts": rad_config.do_sfcperts,
        "pertalb": rad_config.pertalb,
        "do_only_clearsky_rad": rad_config.do_only_clearsky_rad,
        "swhtr": rad_config.swhtr,
        "solcon": rad_config.solcon,
        "lprnt": rad_config.lprnt,
        "lwhtr": rad_config.lwhtr,
        "lssav": rad_config.lssav,
    }
    for tracer_name, index in tracer_inds.items():
        if tracer_name in tracer_name_mapping:
            model[tracer_name_mapping[tracer_name]] = index
    model["ntrac"] = max(tracer_inds.values())
    return model


def _solar_hour(time: cftime.DatetimeJulian) -> float:
    return (
        time.hour
        + time.minute / MINUTES_PER_HOUR
        + time.second / (MINUTES_PER_HOUR * SECONDS_PER_MINUTE)
    )


def get_statein(
    state: State,
    tracer_inds: Mapping[str, int],
    ivflip: int,
    unstacked_dim="z",
    base_state_names: Sequence[str] = STATE_NAMES_IN,
) -> Mapping[str, np.ndarray]:
    tracer_names = list(tracer_inds.keys())
    state_names = list(base_state_names) + tracer_names
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

    statein = {
        "prsi": pi.values,
        "prsl": pm.values,
        "tgrs": temperature.values,
        "prslk": exn.values,
        "qgrs": tracer_array,
    }
    if ivflip == 1:
        statein = {name: np.flip(statein[name], axis=1) for name in statein}
    return statein


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


def get_grid(
    state: State, grid_names: Sequence[str] = GRID_NAMES_IN
) -> Tuple[Mapping[str, np.ndarray], xr.DataArray]:
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


def get_sfcprop(
    state: State, sfc_var_mapping: Mapping[str, str] = SFC_NAMES_IN_MAPPING
) -> Mapping[str, np.ndarray]:
    sfc = xr.Dataset({v: state[k] for k, v in list(sfc_var_mapping.items())})
    # we only want the first of "hprime" variables
    sfc["hprime"] = sfc["hprime"].isel({"orographic_variable": 0})
    stacked_sfc = _stack(sfc)
    return {name: stacked_sfc[name].values for name in stacked_sfc.data_vars}


def postprocess_out(
    out: Tuple[Mapping[str, np.ndarray], ...],
    out_rename: Mapping[str, str] = RENAME_OUT,
) -> Mapping[str, np.ndarray]:
    flattened = []
    for out_dict in out:
        flattened.append(_flatten(out_dict))
    merged = _merge(*flattened)
    return {new_name: merged[old_name] for old_name, new_name in out_rename.items()}


def _merge(*out_dicts: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    """Merge flattened output dictionaries, which should not have name clashes"""
    merged: MutableMapping[str, np.ndarray] = {}
    names: Set[str] = set()
    for out in out_dicts:
        _names = set(out.keys())
        overlapping = names & _names
        if len(overlapping) > 0:
            raise ValueError("Overlapping outputs found: {overlapping}")
        else:
            names = names | _names
        merged.update(**out)
    return merged


def _flatten(in_mapping: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    """Flatten output dictionaries via key concatenation"""
    flattened = {}
    for k, v in in_mapping.items():
        if isinstance(v, dict):
            flattened.update(
                {
                    "_".join([k, nested_k]): nested_v
                    for nested_k, nested_v in _flatten(v).items()
                }
            )
        elif isinstance(v, np.ndarray):
            flattened[k] = v
        else:
            raise TypeError("Nested dict contains non-numpy array items.")
    return flattened
