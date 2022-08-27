import dataclasses
from typing import Optional, Mapping, Any, Literal, Tuple, Sequence
import cftime
import numpy as np
import xarray as xr
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    open_model,
    MultiModelAdapter,
)
from runtime.derived_state import DerivedFV3State
from radiation import RadiationDriver, getdata, NGPTSW, NGPTLW
from vcm.calc.thermo.vertically_dependent import (
    pressure_at_interface,
    pressure_at_midpoint_log,
)
from vcm.calc.thermo.constants import _SPECIFIC_HEAT_CONST_PRESSURE, _RDGAS

P_REF = 1.0e5


DEFAULT_RAD_CONFIG = {
    "imp_physics": 11,
    "iemsflg": 1,  # surface emissivity control flag
    "ioznflg": 7,  # ozone data source control flag
    "ictmflg": 1,  # data ic time/date control flag
    "isolar": 2,  # solar constant control flag
    "ico2flg": 2,  # co2 data source control flag
    "iaerflg": 111,  # volcanic aerosols
    "ialbflg": 1,  # surface albedo control flag
    "icldflg": 1,
    "ivflip": 1,  # vertical index direction control flag
    "iovrsw": 1,  # cloud overlapping control flag for sw
    "iovrlw": 1,  # cloud overlapping control flag for lw
    "isubcsw": 2,  # sub-column cloud approx flag in sw radiation
    "isubclw": 2,  # sub-column cloud approx flag in lw radiation
    "lcrick": False,  # control flag for eliminating CRICK
    "lcnorm": False,  # control flag for in-cld condensate
    "lnoprec": False,  # precip effect on radiation flag (ferrier microphysics)
    "iswcliq": 1,  # optical property for liquid clouds for sw
    "lsswr": True,
    "lslwr": True,
    "nfxr": 45,
    "ncld": 5,
    "ncnd": 5,
    "solhr": 0.0,
    "lgfdlmprad": False,
    "uni_cld": False,
    "effr_in": False,
    "indcld": -1,
    "num_p3d": 1,
    "npdf3d": 0,
    "ncnvcld3d": 0,
    "lmfdeep2": True,
    "lmfshal": True,
    "sup": 1.0,
    "kdt": 1,
    "do_sfcperts": False,
    "pertalb": [[-999.0], [-999.0], [-999.0], [-999.0], [-999.0]],
    "do_only_clearsky_rad": False,
    "swhtr": True,
    "solcon": 1320.8872136343873,
    "lprnt": False,
    "lwhtr": True,
    "lssav": True,
}


@dataclasses.dataclass
class RadiationWrapperConfig:
    """"""

    kind: Literal["python"]
    input_model: Optional[MachineLearningConfig] = None
    offline: bool = True


class RadiationWrapper:
    def __init__(
        self,
        offline: bool,
        input_model: Optional[MultiModelAdapter],
        driver: RadiationDriver,
    ):
        self._offline: bool = offline
        self._input_model: Optional[MultiModelAdapter] = input_model
        self._driver: RadiationDriver = driver
        self._rad_config: Optional[Mapping[str, Any]] = None

    @classmethod
    def from_config(cls, config: RadiationWrapperConfig) -> "RadiationWrapper":
        if config.input_model:
            model: Optional[MultiModelAdapter] = open_model(config.input_model)
        else:
            model = None
        driver = RadiationDriver()
        return cls(config.offline, model, driver)

    def rad_init(
        self,
        rank: int,
        physics_namelist: dict,
        forcing_dir: str = "./data/forcing",
        fv_core_dir: str = "./INPUT/",
        default_rad_config: dict = DEFAULT_RAD_CONFIG,
    ) -> None:
        """Initialize the radiation driver"""
        self._rad_config = self._get_rad_config(physics_namelist, default_rad_config)
        sigma = getdata.sigma(fv_core_dir)
        nlay = len(sigma) - 1
        aerosol_data = getdata.aerosol(forcing_dir)
        sfc_filename, sfc_data = getdata.sfc(forcing_dir)
        solar_filename, _ = getdata.astronomy(forcing_dir, self._rad_config["isolar"])

        self._driver.radinit(
            sigma,
            nlay,
            self._rad_config["imp_physics"],
            rank,
            self._rad_config["iemsflg"],
            self._rad_config["ioznflg"],
            self._rad_config["ictmflg"],
            self._rad_config["isolar"],
            self._rad_config["ico2flg"],
            self._rad_config["iaerflg"],
            self._rad_config["ialbflg"],
            self._rad_config["icldflg"],
            self._rad_config["ivflip"],
            self._rad_config["iovrsw"],
            self._rad_config["iovrlw"],
            self._rad_config["isubcsw"],
            self._rad_config["isubclw"],
            self._rad_config["lcrick"],
            self._rad_config["lcnorm"],
            self._rad_config["lnoprec"],
            self._rad_config["iswcliq"],
            aerosol_data,
            solar_filename,
            sfc_filename,
            sfc_data,
        )

    @staticmethod
    def _get_rad_config(physics_namelist, default_rad_config) -> Mapping[str, Any]:
        """Generate radiation config from fv3gfs namelist to ensure
        identical. Additional values from hardcoded defaults.
        """
        rad_config = {}
        rad_config["imp_physics"] = physics_namelist["imp_physics"]
        rad_config["iemsflg"] = physics_namelist["iems"]
        rad_config["ioznflg"] = default_rad_config["ioznflg"]
        rad_config["ictmflg"] = default_rad_config["ictmflg"]
        rad_config["isolar"] = physics_namelist["isol"]
        rad_config["ico2flg"] = physics_namelist["ico2"]
        rad_config["iaerflg"] = physics_namelist["iaer"]
        rad_config["ialbflg"] = physics_namelist["ialb"]
        rad_config["icldflg"] = default_rad_config["icldflg"]
        rad_config["ivflip"] = default_rad_config["ivflip"]
        rad_config["iovrsw"] = default_rad_config["iovrsw"]
        rad_config["iovrlw"] = default_rad_config["iovrlw"]
        rad_config["isubcsw"] = physics_namelist["isubc_sw"]
        rad_config["isubclw"] = physics_namelist["isubc_lw"]
        rad_config["lcrick"] = default_rad_config["lcrick"]
        rad_config["lcnorm"] = default_rad_config["lcnorm"]
        rad_config["lnoprec"] = default_rad_config["lnoprec"]
        rad_config["iswcliq"] = default_rad_config["iswcliq"]
        rad_config["fhswr"] = physics_namelist["fhswr"]
        rad_config["lsswr"] = default_rad_config["lsswr"]
        rad_config["lslwr"] = default_rad_config["lslwr"]
        rad_config["solhr"] = default_rad_config["solhr"]
        rad_config["nfxr"] = default_rad_config["nfxr"]
        rad_config["ncld"] = physics_namelist["ncld"]
        rad_config["ncnd"] = physics_namelist["ncld"]
        rad_config["fhswr"] = physics_namelist["fhswr"]
        rad_config["fhlwr"] = physics_namelist["fhlwr"]
        rad_config["lgfdlmprad"] = default_rad_config["lgfdlmprad"]
        rad_config["uni_cld"] = default_rad_config["uni_cld"]
        rad_config["effr_in"] = default_rad_config["effr_in"]
        rad_config["indcld"] = default_rad_config["indcld"]
        rad_config["num_p3d"] = default_rad_config["num_p3d"]
        rad_config["npdf3d"] = default_rad_config["npdf3d"]
        rad_config["ncnvcld3d"] = default_rad_config["ncnvcld3d"]
        rad_config["lmfdeep2"] = default_rad_config["lmfdeep2"]
        rad_config["lmfshal"] = default_rad_config["lmfshal"]
        rad_config["sup"] = default_rad_config["sup"]
        rad_config["kdt"] = default_rad_config["kdt"]
        rad_config["do_sfcperts"] = default_rad_config["do_sfcperts"]
        rad_config["pertalb"] = default_rad_config["pertalb"]
        rad_config["do_only_clearsky_rad"] = default_rad_config["do_only_clearsky_rad"]
        rad_config["swhtr"] = physics_namelist["swhtr"]
        rad_config["solcon"] = default_rad_config["solcon"]
        rad_config["lprnt"] = default_rad_config["lprnt"]
        rad_config["lwhtr"] = physics_namelist["lwhtr"]
        rad_config["lssav"] = default_rad_config["lssav"]

        return rad_config

    def rad_update(
        self,
        time: cftime.DatetimeJulian,
        dt_atmos: float,
        forcing_dir: str = "./data/forcing",
    ) -> None:
        """Update the radiation driver's time-varying parameters"""
        if self._rad_config is None:
            raise ValueError(
                "Radiation driver not initialized. `.rad_init` must be called "
                "before `.rad_update`."
            )
        idat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, 0, 0]
        )
        jdat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, 0, 0]
        )
        fhswr = np.array(float(self._rad_config["fhswr"]))
        dt_atmos = np.array(float(dt_atmos))
        aerosol_data = getdata.aerosol(forcing_dir)
        _, solar_data = getdata.astronomy(forcing_dir, self._rad_config["isolar"])
        gas_data = getdata.gases(forcing_dir, self._rad_config["ictmflg"])
        slag, sdec, cdec, solcon = self._driver.radupdate(
            idat,
            jdat,
            fhswr,
            dt_atmos,
            self._rad_config["lsswr"],
            aerosol_data["kprfg"],
            aerosol_data["idxcg"],
            aerosol_data["cmixg"],
            aerosol_data["denng"],
            aerosol_data["cline"],
            solar_data,
            gas_data,
        )

    def rad_compute(
        self,
        state: DerivedFV3State,
        tracer_metadata: Mapping[str, Any],
        rank: int,
        lookup_dir: str = "./data/lookup",
    ):
        if self._rad_config is None:
            raise ValueError(
                "Radiation driver not initialized. `.rad_init` must be called "
                "before `.rad_compute`."
            )
        statein = _statein(state, tracer_metadata, self._rad_config["ivflip"])
        grid, coords = _grid(state)
        sfcprop = _sfcprop(state)
        ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
        model = _model(self._rad_config, tracer_metadata, nz, rank)
        random_numbers = getdata.random(ncolumns, nz, NGPTSW, NGPTLW)
        lw_lookup = getdata.lw(lookup_dir)
        sw_lookup = getdata.sw(lookup_dir)
        out = self._driver._GFS_radiation_driver(
            model, statein, sfcprop, grid, random_numbers, lw_lookup, sw_lookup
        )
        return _unstack(_rename_out(out), coords)


# These could be refactored to the radiation package's getdata module,
# since they're pretty specific to the radiation port's demands

TRACER_NAME_MAPPING = {  # this is specific to GFS physics
    "cloud_water_mixing_ratio": "ntcw",
    "rain_mixing_ratio": "ntrw",
    "cloud_ice_mixing_ratio": "ntiw",
    "snow_mixing_ratio": "ntsw",
    "graupel_mixing_ratio": "ntgl",
    "ozone_mixing_ratio": "ntoz",
    "cloud_amount": "ntclamt",
}


def _model(
    rad_config: Mapping[str, Any],
    tracer_metadata: Mapping[str, Any],
    nz: int,
    rank: int,
) -> Mapping[str, Any]:
    model = {
        "me": rank,
        "levs": nz,
        "levr": nz,
        "nfxr": rad_config["nfxr"],
        "ncld": rad_config["ncld"],
        "ncnd": rad_config["ncnd"],
        "fhswr": rad_config["fhswr"],
        "fhlwr": rad_config["fhlwr"],
        "solhr": rad_config["solhr"],
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
    tracer_inds = []
    for tracer_name, metadata in tracer_metadata.items():
        if tracer_name in TRACER_NAME_MAPPING:
            tracer_inds.append(metadata["i_tracer"])
            model[TRACER_NAME_MAPPING[tracer_name]] = metadata["i_tracer"]
    model["ntrac"] = max(tracer_inds)
    return model


def _statein(
    state: DerivedFV3State,
    tracer_metadata: Mapping[str, Any],
    ivflip: int,
    unstacked_dim="z",
) -> Mapping[str, np.ndarray]:
    state_names = ["pressure_thickness_of_atmospheric_layer", "air_temperature"]
    tracer_names = list(tracer_metadata.keys())
    tracer_inds = [
        tracer_metadata[name]["i_tracer"] - 1
        for name in tracer_names  # fortran to python index
    ]
    state_names += tracer_names
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
    for n, tracer_name in zip(tracer_inds, tracer_names):
        tracer_array[:, :, n] = stacked[tracer_name].values

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


def _unstack(
    data: Mapping[str, np.ndarray],
    coords: xr.DataArray,
    sample_dim: str = "column",
    unstacked_dim: str = "z",
) -> Mapping[str, xr.DataArray]:
    out = {}
    for name, arr in data.items():
        if arr.ndim == 1:
            da = xr.DataArray(arr, dims=[sample_dim], coords={sample_dim: coords})
        elif arr.ndim == 2:
            da = xr.DataArray(
                arr, dims=[sample_dim, unstacked_dim], coords={sample_dim: coords}
            )
        out[name] = da.unstack(dim=sample_dim)
    return out


def _grid(state: DerivedFV3State) -> Tuple[Mapping[str, np.ndarray], xr.DataArray]:
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


def _sfcprop(
    state: DerivedFV3State, sfc_var_mapping: Mapping[str, str] = SFC_VAR_MAPPING
) -> Mapping[str, np.ndarray]:
    sfc = xr.Dataset({k: state[v] for k, v in list(sfc_var_mapping.items())})
    stacked_sfc = _stack(sfc)
    return {name: stacked_sfc[name].values for name in stacked_sfc.data_vars}


OUT_NAMES = [
    "clear_sky_downward_longwave_flux_at_surface",
    "clear_sky_downward_shortwave_flux_at_surface",
    "clear_sky_upward_longwave_flux_at_surface",
    "clear_sky_upward_shortwave_flux_at_surface",
    "clear_sky_upward_longwave_flux_at_top_of_atmosphere",
    "clear_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_downward_longwave_flux_at_surface",
    "total_sky_downward_shortwave_flux_at_surface",
    "total_sky_upward_longwave_flux_at_surface",
    "total_sky_upward_shortwave_flux_at_surface",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
]


def _rename_out(
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
