import dataclasses
import os
from typing import (
    Optional,
    MutableMapping,
    Mapping,
    Any,
    Literal,
    Tuple,
    Sequence,
    Hashable,
)
import tarfile
from mpi4py import MPI
import cftime
from datetime import timedelta
import numpy as np
import xarray as xr
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    open_model,
    MultiModelAdapter,
    predict,
)
from runtime.types import State, Diagnostics
from runtime.derived_state import DerivedFV3State
from runtime.steppers.radiation._config import (
    get_rad_config,
    LOOKUP_DATA_PATH,
    FORCING_DATA_PATH,
)
from radiation import RadiationDriver, getdata, NGPTSW, NGPTLW
from vcm.calc.thermo.vertically_dependent import (
    pressure_at_interface,
    pressure_at_midpoint_log,
)
from vcm.cloud import get_fs
from vcm.calc.thermo.constants import _SPECIFIC_HEAT_CONST_PRESSURE, _RDGAS

P_REF = 1.0e5
MINUTES_PER_HOUR = 60.0
SECONDS_PER_MINUTE = 60.0


@dataclasses.dataclass
class RadiationConfig:
    """"""

    kind: Literal["python"]
    input_model: Optional[MachineLearningConfig] = None


class RadiationStepper:
    def __init__(
        self,
        driver: RadiationDriver,
        rad_config: MutableMapping[Hashable, Any],
        comm: MPI.COMM_WORLD,
        input_model: Optional[MultiModelAdapter],
    ):
        self._driver: RadiationDriver = driver
        self._rad_config: MutableMapping[Hashable, Any] = rad_config
        self._comm: MPI.COMM_WORLD = comm
        self._input_model: Optional[MultiModelAdapter] = input_model
        self._download_radiation_assets()
        self._init_driver()

    @classmethod
    def from_config(
        cls,
        config: RadiationConfig,
        comm: MPI.COMM_WORLD,
        physics_namelist: Mapping[Hashable, Any],
    ) -> "RadiationStepper":
        rad_config = get_rad_config(physics_namelist)
        if config.input_model:
            model: Optional[MultiModelAdapter] = open_model(config.input_model)
        else:
            model = None
        return cls(RadiationDriver(), rad_config, comm, model)

    def _download_radiation_assets(
        self,
        lookup_data_path: str = LOOKUP_DATA_PATH,
        forcing_data_path: str = FORCING_DATA_PATH,
        lookup_local_dir: str = "./rad_data/lookup/",
        forcing_local_dir: str = "./rad_data/forcing/",
    ) -> None:
        """Gets lookup tables and forcing needed for the radiation scheme.
        TODO: make scheme able to read existing forcing; make lookup data part of
        writing a run directory?
        """
        if self._comm.rank == 0:
            for target, local in zip(
                (lookup_data_path, forcing_data_path),
                (lookup_local_dir, forcing_local_dir),
            ):
                _get_remote_tar_data(target, local)
        self._comm.barrier()
        self._lookup_local_dir = lookup_local_dir
        self._forcing_local_dir = forcing_local_dir

    def _init_driver(self, fv_core_dir: str = "./INPUT/"):
        """Initialize the radiation driver"""
        sigma = getdata.sigma(fv_core_dir)
        nlay = len(sigma) - 1
        aerosol_data = getdata.aerosol(self._forcing_local_dir)
        sfc_filename, sfc_data = getdata.sfc(self._forcing_local_dir)
        solar_filename, _ = getdata.astronomy(
            self._forcing_local_dir, self._rad_config["isolar"]
        )
        self._driver.radinit(
            sigma,
            nlay,
            self._rad_config["imp_physics"],
            self._comm.rank,
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

    def __call__(
        self,
        state: State,
        tracer_metadata: Mapping[str, Any],
        time: cftime.DatetimeJulian,
        dt_atmos: float,
    ):

        self._rad_update(time, dt_atmos)
        diagnostics = self._rad_compute(state, tracer_metadata, time, dt_atmos)
        return {}, diagnostics, {}

    def _rad_update(self, time: cftime.DatetimeJulian, dt_atmos: float) -> None:
        """Update the radiation driver's time-varying parameters"""
        # idat is supposed to be model initalization time but is unused w/ current flags
        idat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, time.second, 0]
        )
        jdat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, time.second, 0]
        )
        fhswr = np.array(float(self._rad_config["fhswr"]))
        dt_atmos = np.array(float(dt_atmos))
        aerosol_data = getdata.aerosol(self._forcing_local_dir)
        _, solar_data = getdata.astronomy(
            self._forcing_local_dir, self._rad_config["isolar"]
        )
        gas_data = getdata.gases(self._forcing_local_dir, self._rad_config["ictmflg"])
        self._driver.radupdate(
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

    def _rad_compute(
        self,
        state: State,
        tracer_metadata: Mapping[str, Any],
        time: cftime.DatetimeJulian,
        dt_atmos: float,
    ) -> Diagnostics:
        """Compute the radiative fluxes"""
        if self._input_model is not None:
            predictions = predict(self._input_model, state)
            state = UpdatedState(state, predictions)
        statein = _statein(state, tracer_metadata, self._rad_config["ivflip"])
        grid, coords = _grid(state)
        sfcprop = _sfcprop(state)
        ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
        model = _model(
            self._rad_config, tracer_metadata, time, dt_atmos, nz, self._comm.rank
        )
        random_numbers = getdata.random(ncolumns, nz, NGPTSW, NGPTLW)
        lw_lookup = getdata.lw(self._lookup_local_dir)
        sw_lookup = getdata.sw(self._lookup_local_dir)
        out = self._driver._GFS_radiation_driver(
            model, statein, sfcprop, grid, random_numbers, lw_lookup, sw_lookup
        )
        return _unstack(_rename_out(out), coords)


def _get_remote_tar_data(remote_filepath: str, local_dir: str):
    os.makedirs(local_dir)
    fs = get_fs(remote_filepath)
    fs.get(remote_filepath, local_dir)
    local_filepath = os.path.join(local_dir, os.path.basename(remote_filepath))
    tarfile.open(local_filepath).extractall(path=local_dir)


class UpdatedState(DerivedFV3State):
    def __init__(self, state: State, predictions: State):
        self._state = state
        self._predictions = predictions

    def __getitem__(self, key: Hashable) -> xr.DataArray:
        return self._predictions.get(key, self._state[key])


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
    rad_config: MutableMapping[Hashable, Any],
    tracer_metadata: Mapping[str, Any],
    time: cftime.DatetimeJulian,
    dt_atmos: float,
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
    tracer_inds = []
    for tracer_name, metadata in tracer_metadata.items():
        if tracer_name in TRACER_NAME_MAPPING:
            tracer_inds.append(metadata["i_tracer"])
            model[TRACER_NAME_MAPPING[tracer_name]] = metadata["i_tracer"]
    model["ntrac"] = max(tracer_inds)
    return model


def _solar_hour(time: cftime.DatetimeJulian) -> float:
    return (
        time.hour
        + time.minute / MINUTES_PER_HOUR
        + time.second / (MINUTES_PER_HOUR * SECONDS_PER_MINUTE)
    )


def _statein(
    state: State, tracer_metadata: Mapping[str, Any], ivflip: int, unstacked_dim="z",
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


def _grid(state: State) -> Tuple[Mapping[str, np.ndarray], xr.DataArray]:
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
