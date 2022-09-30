from typing import MutableMapping, Mapping, Hashable, Sequence, Optional

try:
    from mpi4py import MPI
except ImportError:
    pass
from datetime import timedelta
import cftime
import numpy as np
import xarray as xr
from radiation.config import (
    RadiationConfig,
    GFSPhysicsControl,
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
MINUTES_PER_HOUR: float = 60.0
SECONDS_PER_MINUTE: float = 60.0


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
        tracer_inds: Mapping[str, int],
    ):
        self._driver: RadiationDriver = RadiationDriver()
        self._rad_config: RadiationConfig = rad_config
        self._comm: "MPI.COMM_WORLD" = comm
        self._timestep: float = timestep
        self._tracer_inds: Mapping[str, int] = tracer_inds

        self._solar_data: Optional[xr.Dataset] = None
        self._aerosol_data: Mapping = dict()
        self._sfc_data: Optional[xr.Dataset] = None
        self._gas_data: Mapping = dict()
        self._sw_lookup: Mapping = dict()
        self._lw_lookup: Mapping = dict()

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

    def __call__(
        self, time: cftime.DatetimeJulian, state: State,
    ):
        self._rad_update(time, self._timestep)
        diagnostics = self._rad_compute(state, time)
        return diagnostics

    def _rad_update(self, time: cftime.DatetimeJulian, dt_atmos: float) -> None:
        """Update the radiation driver's time-varying parameters"""
        # idat is supposed to be model initalization time but is unused w/ current flags
        idat = np.array(
            [time.year, time.month, time.day, 0, time.hour, time.minute, time.second, 0]
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
        # todo: update wrapper with fix to avoid having to be one timestep off to
        # compute solar hour, see https://github.com/ai2cm/fv3net/issues/2071
        solhr = _solar_hour(time - timedelta(seconds=self._timestep))
        statein = get_statein(state, self._tracer_inds, self._rad_config.ivflip)
        grid, coords = get_grid(state)
        sfcprop = get_sfcprop(state)
        ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
        gfs_physics_control = self._get_gfs_physics_control(time, nz)
        random_numbers = io.generate_random_numbers(ncolumns, nz, NGPTSW, NGPTLW)
        out = self._driver._GFS_radiation_driver(
            gfs_physics_control,
            self._driver.solar_constant,
            solhr,
            statein,
            sfcprop,
            grid,
            random_numbers,
            self._lw_lookup,
            self._sw_lookup,
        )
        out = postprocess_out(out)
        return unstack(out, coords)

    def _get_gfs_physics_control(
        self,
        time: cftime.DatetimeJulian,
        nz: int,
        tracer_name_mapping: Mapping[str, str] = TRACER_NAMES_IN_MAPPING,
    ) -> GFSPhysicsControl:
        gfs_physics_control = self._rad_config.gfs_physics_control
        gfs_physics_control.levs = nz
        gfs_physics_control.levr = nz
        for tracer_name, index in self._tracer_inds.items():
            if hasattr(gfs_physics_control, tracer_name_mapping[tracer_name]):
                setattr(gfs_physics_control, tracer_name_mapping[tracer_name], index)
        gfs_physics_control.ntrac = max(self._tracer_inds.values())
        return gfs_physics_control


def _solar_hour(time: cftime.DatetimeJulian) -> float:
    return (
        time.hour
        + time.minute / MINUTES_PER_HOUR
        + time.second / (MINUTES_PER_HOUR * SECONDS_PER_MINUTE)
    )
