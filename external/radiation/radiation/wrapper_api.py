from typing import MutableMapping, Mapping, Hashable, Sequence, Optional

try:
    from mpi4py import MPI
except ImportError:
    pass
import cftime
import numpy as np
import xarray as xr
from radiation.config import (
    RadiationConfig,
    LOOKUP_DATA_PATH,
    FORCING_DATA_PATH,
)
from radiation.radsw import ngptsw as NGPTSW
from radiation.radlw import ngptlw as NGPTLW
from radiation.radiation_driver import RadiationDriver
from radiation import io
from radiation.preprocessing import (
    get_statein,
    get_sfcprop,
    get_grid,
    postprocess_out,
    unstack,
    BASE_INPUT_VARIABLE_NAMES,
    OUTPUT_VARIABLE_NAMES,
)

TRACER_NAMES_IN_MAPPING: Mapping[str, str] = {
    "cloud_water_mixing_ratio": "ntcw",
    "rain_mixing_ratio": "ntrw",
    "cloud_ice_mixing_ratio": "ntiw",
    "snow_mixing_ratio": "ntsw",
    "graupel_mixing_ratio": "ntgl",
    "ozone_mixing_ratio": "ntoz",
    "cloud_amount": "ntclamt",
}
SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0


State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


class Radiation:

    _base_input_variables: Sequence[str] = BASE_INPUT_VARIABLE_NAMES
    output_variables: Sequence[str] = OUTPUT_VARIABLE_NAMES

    def __init__(
        self,
        rad_config: RadiationConfig,
        comm: "MPI.COMM_WORLD",
        timestep: float,
        init_time: cftime.DatetimeJulian,
        tracer_inds: Mapping[str, int],
    ):
        self._driver: RadiationDriver = RadiationDriver()
        self._rad_config: RadiationConfig = rad_config
        self._comm: "MPI.COMM_WORLD" = comm
        self._timestep: float = timestep
        self._init_time: cftime.DatetimeJulian = init_time
        self._tracer_inds: Mapping[str, int] = tracer_inds

        self._solar_data: Optional[xr.Dataset] = None
        self._aerosol_data: Mapping = dict()
        self._sfc_data: Optional[xr.Dataset] = None
        self._gas_data: Mapping = dict()
        self._sw_lookup: Mapping = dict()
        self._lw_lookup: Mapping = dict()

        self._cached: Diagnostics = {}

        self._download_radiation_assets()
        self._init_driver()

    @property
    def input_variables(self):
        return self._base_input_variables + list(self._tracer_inds.keys())

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
                io.get_remote_tar_data(target, local)
        self._comm.barrier()
        self._lookup_local_dir = lookup_local_dir
        self._forcing_local_dir = forcing_local_dir

    def _init_driver(self, fv_core_dir: str = "./INPUT/"):
        """Initialize the radiation driver"""
        sigma = io.load_sigma(fv_core_dir)
        nlay = len(sigma) - 1
        self._aerosol_data = io.load_aerosol(self._forcing_local_dir)
        sfc_filename, self._sfc_data = io.load_sfc(self._forcing_local_dir)
        solar_filename, self._solar_data = io.load_astronomy(
            self._forcing_local_dir, self._rad_config.isolar
        )
        self._gas_data = io.load_gases(
            self._forcing_local_dir, self._rad_config.ictmflg
        )
        self._init_gfs_physics_control(nlay)
        self._driver.radinit(
            sigma,
            nlay,
            self._rad_config.gfs_physics_control.imp_physics,
            self._comm.rank,
            self._rad_config.iemsflg,
            self._rad_config.ioznflg,
            self._rad_config.ictmflg,
            self._rad_config.isolar,
            self._rad_config.ico2flg,
            self._rad_config.iaerflg,
            self._rad_config.ialbflg,
            self._rad_config.icldflg,
            self._rad_config.ivflip,
            self._rad_config.iovrsw,
            self._rad_config.iovrlw,
            self._rad_config.isubcsw,
            self._rad_config.isubclw,
            self._rad_config.lcrick,
            self._rad_config.lcnorm,
            self._rad_config.lnoprec,
            self._rad_config.iswcliq,
            self._aerosol_data,
            solar_filename,
            sfc_filename,
            self._sfc_data,
        )
        self._lw_lookup = io.load_lw(self._lookup_local_dir)
        self._sw_lookup = io.load_sw(self._lookup_local_dir)

    def _init_gfs_physics_control(
        self, nz: int, tracer_name_mapping: Mapping[str, str] = TRACER_NAMES_IN_MAPPING,
    ) -> None:
        gfs_physics_control = self._rad_config.gfs_physics_control
        gfs_physics_control.levs = nz
        gfs_physics_control.levr = nz
        for tracer_name, index in self._tracer_inds.items():
            if tracer_name in tracer_name_mapping and hasattr(
                gfs_physics_control, tracer_name_mapping[tracer_name]
            ):
                setattr(gfs_physics_control, tracer_name_mapping[tracer_name], index)
        gfs_physics_control.ntrac = max(self._tracer_inds.values())
        gfs_physics_control.nsswr = int(gfs_physics_control.fhswr / self._timestep)
        gfs_physics_control.nslwr = int(gfs_physics_control.fhlwr / self._timestep)

    def __call__(
        self, time: cftime.DatetimeJulian, state: State,
    ):
        self._set_compute_flags(time)
        if (
            self._rad_config.gfs_physics_control.lslwr
            or self._rad_config.gfs_physics_control.lsswr
        ):
            self._rad_update(time, self._timestep)
        diagnostics = self._rad_compute(state, time)
        return diagnostics

    def _set_compute_flags(self, time: cftime.DatetimeJulian) -> None:
        timestep_index = _get_forecast_time_index(time, self._init_time, self._timestep)
        sw_compute_period = self._rad_config.gfs_physics_control.nsswr
        if sw_compute_period is None:
            raise ValueError("GFS physics control nsswr not set.")
        self._rad_config.gfs_physics_control.lsswr = _is_compute_timestep(
            timestep_index, sw_compute_period
        )
        lw_compute_period = self._rad_config.gfs_physics_control.nslwr
        if lw_compute_period is None:
            raise ValueError("GFS physics control nslwr not set.")
        self._rad_config.gfs_physics_control.lslwr = _is_compute_timestep(
            timestep_index, lw_compute_period
        )

    def _rad_update(self, time: cftime.DatetimeJulian, dt_atmos: float) -> None:
        """Update the radiation driver's time-varying parameters"""

        idat = np.array(
            [
                self._init_time.year,
                self._init_time.month,
                self._init_time.day,
                0,
                self._init_time.hour,
                self._init_time.minute,
                self._init_time.second,
                0,
            ]
        )
        jdat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, time.second, 0]
        )
        fhswr = np.array(float(self._rad_config.gfs_physics_control.fhswr))
        dt_atmos = np.array(float(dt_atmos))
        self._driver.radupdate(
            idat,
            jdat,
            fhswr,
            dt_atmos,
            self._rad_config.gfs_physics_control.lsswr,
            self._aerosol_data["kprfg"],
            self._aerosol_data["idxcg"],
            self._aerosol_data["cmixg"],
            self._aerosol_data["denng"],
            self._aerosol_data["cline"],
            self._solar_data,
            self._gas_data,
            self._comm.rank,
        )

    def _rad_compute(self, state: State, time: cftime.DatetimeJulian,) -> Diagnostics:
        """Compute the radiative fluxes"""
        if (
            self._rad_config.gfs_physics_control.lsswr
            or self._rad_config.gfs_physics_control.lslwr
        ):
            solhr = self._solar_hour(time)
            statein = get_statein(state, self._tracer_inds, self._rad_config.ivflip)
            grid, coords = get_grid(state)
            sfcprop = get_sfcprop(state)
            ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
            random_numbers = io.generate_random_numbers(ncolumns, nz, NGPTSW, NGPTLW)
            out = self._driver._GFS_radiation_driver(
                self._rad_config.gfs_physics_control,
                self._driver.solar_constant,
                solhr,
                statein,
                sfcprop,
                grid,
                random_numbers,
                self._lw_lookup,
                self._sw_lookup,
            )
            self._cached = unstack(postprocess_out(out), coords)
        return self._cached

    def _solar_hour(self, time: cftime.DatetimeJulian) -> float:
        return _solar_hour(time, self._init_time)


def _get_forecast_time_index(
    current_time: cftime.DatetimeJulian,
    init_time: cftime.DatetimeJulian,
    timestep_seconds: float,
) -> int:
    """Get integer index of forecast time, since initial time """
    forecast_elapsed_seconds = (current_time - init_time).total_seconds()
    return int(forecast_elapsed_seconds / timestep_seconds) + 1


def _is_compute_timestep(timestep_index: int, compute_period: int) -> bool:
    if compute_period == 1:
        return True
    elif timestep_index % compute_period == 1:
        return True
    else:
        return False


def _solar_hour(time: cftime.DatetimeJulian, init_time: cftime.DatetimeJulian) -> float:
    """This follows the Fortran computation that shifts the solar hour if
    initialization is not on the hour. See
    https://github.com/NOAA-GFDL/SHiELD_physics/issues/17, but for
    now we want the port to validate against Fortran."""
    seconds_elapsed = (time - init_time).total_seconds()
    hours_elapsed = seconds_elapsed / SECONDS_PER_HOUR
    return (hours_elapsed + init_time.hour) % HOURS_PER_DAY
