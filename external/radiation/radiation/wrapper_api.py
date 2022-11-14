from typing import MutableMapping, Mapping, Hashable, Sequence, Optional, Any
import dataclasses

try:
    from mpi4py import MPI
except ImportError:
    pass
import cftime
import numpy as np
import xarray as xr
from radiation.config import (
    RadiationConfig,
    GFSPhysicsControlConfig,
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

SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0


State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


@dataclasses.dataclass
class GFSPhysicsControl:

    """Ported version of the dynamic Fortran GFS_physics_control structure
        ('model' in the Fortran radiation code).
    
    Args:
        config: GFS physics control configuration
        ntcw: Tracer index of cloud liquid water.
        ntrw: Tracer index of rain water.
        ntiw: Tracer index of cloud ice water.
        ntsw: Tracer index of snow water.
        ntgl: Tracer index of graupel water.
        ntoz: Tracer index of ozone.
        ntclamt: Tracer index of cloud amount.
        ntrac: Number of tracers.
        nsswr: Integer number of physics timesteps between shortwave radiation
            calculations.
        nslwr: Integer number of physics timesteps between longwave radiation
            calculations.
        lsswr: Time-varying logical flag for SW radiation calculations.
        lslwr: Time-varying logical flag for LW radiation calculations.
    """

    config: GFSPhysicsControlConfig
    ntcw: int
    ntrw: int
    ntiw: int
    ntsw: int
    ntgl: int
    ntoz: int
    ntclamt: int
    ntrac: int
    nsswr: int
    nslwr: int
    lsswr: bool = True
    lslwr: bool = True

    @classmethod
    def from_config(
        cls,
        config: GFSPhysicsControlConfig,
        timestep: float,
        tracer_inds: Mapping[str, int],
    ) -> "GFSPhysicsControl":
        """Create a GFS Physics Control instance from its config
        
        Args:
            config: GFS physics control config object
            timestep: Model physics timestep in seconds
            tracer_inds: Mapping of GFS physics tracer names to their index; can be
                obtained from fv3gfs.wrapper.get_tracer_metadata().
        """

        TRACER_NAMES_TO_NAMELIST_ENTRIES: Mapping[str, str] = {
            "cloud_water_mixing_ratio": "ntcw",
            "rain_mixing_ratio": "ntrw",
            "cloud_ice_mixing_ratio": "ntiw",
            "snow_mixing_ratio": "ntsw",
            "graupel_mixing_ratio": "ntgl",
            "ozone_mixing_ratio": "ntoz",
            "cloud_amount": "ntclamt",
        }

        tracer_ind_kwargs = {}
        field_names = [field.name for field in dataclasses.fields(cls)]
        for tracer_name, index in tracer_inds.items():
            if tracer_name in TRACER_NAMES_TO_NAMELIST_ENTRIES:
                namelist_entry = TRACER_NAMES_TO_NAMELIST_ENTRIES[tracer_name]
                if namelist_entry in field_names:
                    tracer_ind_kwargs[namelist_entry] = index
        ntrac = max(tracer_inds.values())
        nsswr = int(config.fhswr / timestep)
        nslwr = int(config.fhlwr / timestep)
        kwargs: Mapping[str, Any] = dict(
            **tracer_ind_kwargs, config=config, ntrac=ntrac, nsswr=nsswr, nslwr=nslwr
        )
        return cls(**kwargs)

    def __getattr__(self, attr):
        if hasattr(self.config, attr):
            return getattr(self.config, attr)
        elif hasattr(self, attr):
            return getattr(self, attr)
        else:
            raise AttributeError(f"GFSPhysicsControl has no attr: {attr}.")


class Radiation:
    """A wrapper around the Python-ported radiation driver.

    Implements `validate`, `init_driver`, `__call__` methods.
    """

    _base_input_variables: Sequence[str] = BASE_INPUT_VARIABLE_NAMES
    output_variables: Sequence[str] = OUTPUT_VARIABLE_NAMES

    def __init__(
        self,
        rad_config: RadiationConfig,
        comm: "MPI.COMM_WORLD",
        timestep: float,
        init_time: cftime.DatetimeJulian,
        tracer_inds: Mapping[str, int],
        driver: Optional[RadiationDriver] = None,
        lookup_local_dir: str = "./rad_data/lookup/",
        forcing_local_dir: str = "./rad_data/forcing/",
    ):
        """
        Args:
            rad_config: A radiation configuration object.
            comm: MPI.COMM_WORLD.
            timestep: Model physics timestep in seconds.
            init_time: FV3GFS initialization time, which may be the initialization
                time of the initial segment of a restarted run.
            tracer_inds: Mapping of GFS physics tracer names to their index; can be
                obtained from fv3gfs.wrapper.get_tracer_metadata().
            driver: Optional, driver implementing the update and compute methods.
                Call Radiation.init_driver to initialize the driver internally.
            lookup_local_dir: Local path for radiation lookup data storage.
            forcing_local_dir: Local path for radiation forcing data storage.
        """
        self._rad_config: RadiationConfig = rad_config
        self._comm: "MPI.COMM_WORLD" = comm
        self._timestep: float = timestep
        self._init_time: cftime.DatetimeJulian = init_time
        self._tracer_inds: Mapping[str, int] = tracer_inds
        self._gfs_physics_control = GFSPhysicsControl.from_config(
            rad_config.gfs_physics_control_config, timestep, tracer_inds
        )
        self._driver: Optional[RadiationDriver] = driver
        self._forcing_local_dir: str = forcing_local_dir
        self._lookup_local_dir: str = lookup_local_dir
        self._solar_data: Optional[xr.Dataset] = None
        self._aerosol_data: Mapping = dict()
        self._sfc_data: Optional[xr.Dataset] = None
        self._gas_data: Mapping = dict()
        self._sw_lookup: Mapping = dict()
        self._lw_lookup: Mapping = dict()
        self._cached: Diagnostics = {}

    def validate(self):
        """Validate the configuration for the radiation driver"""
        if self._comm.rank == 0:
            RadiationDriver.validate(
                self._rad_config.isolar,
                self._rad_config.ictmflg,
                self._rad_config.iovrsw,
                self._rad_config.iovrlw,
                self._rad_config.isubcsw,
                self._rad_config.isubclw,
                self._rad_config.iaerflg,
                self._rad_config.ioznflg,
                self._rad_config.ico2flg,
                self._rad_config.ialbflg,
                self._rad_config.iemsflg,
                self._gfs_physics_control.imp_physics,
                self._rad_config.icldflg,
                self._rad_config.lcrick,
                self._rad_config.lcnorm,
                self._rad_config.lnoprec,
            )

    def init_driver(self):
        """Download necessary data and initialize the driver object"""
        self._download_radiation_assets()
        self._driver = self._init_driver()

    @property
    def input_variables(self):
        """List of state variable names needed to call the radiation driver."""
        return self._base_input_variables + list(self._tracer_inds.keys())

    def _download_radiation_assets(
        self,
        lookup_data_path: str = LOOKUP_DATA_PATH,
        forcing_data_path: str = FORCING_DATA_PATH,
    ) -> None:
        """Gets lookup tables and forcing needed for the radiation scheme.
        TODO: make scheme able to read existing forcing; make lookup data part of
        writing a run directory?
        """
        if self._comm.rank == 0:
            for target, local in zip(
                (lookup_data_path, forcing_data_path),
                (self._lookup_local_dir, self._forcing_local_dir),
            ):
                io.get_remote_tar_data(target, local)
        self._comm.barrier()

    def _init_driver(self, fv_core_dir: str = "./INPUT/"):
        sigma = io.load_sigma(fv_core_dir)
        nlay = len(sigma) - 1
        self._aerosol_data = io.load_aerosol(self._forcing_local_dir)
        self._sfc_data = io.load_sfc(self._forcing_local_dir)
        self._solar_data = io.load_astronomy(
            self._forcing_local_dir, self._rad_config.isolar
        )
        self._gas_data = io.load_gases(
            self._forcing_local_dir, self._rad_config.ictmflg
        )
        self._lw_lookup = io.load_lw(self._lookup_local_dir)
        self._sw_lookup = io.load_sw(self._lookup_local_dir)
        return RadiationDriver(
            sigma,
            nlay,
            self._comm.rank,
            self._rad_config.iemsflg,
            self._rad_config.ioznflg,
            self._rad_config.ictmflg,
            self._rad_config.isolar,
            self._rad_config.ico2flg,
            self._rad_config.iaerflg,
            self._rad_config.ialbflg,
            self._rad_config.ivflip,
            self._rad_config.iovrsw,
            self._rad_config.iovrlw,
            self._rad_config.isubcsw,
            self._rad_config.isubclw,
            self._rad_config.lcnorm,
            self._aerosol_data,
            self._sfc_data,
        )

    def __call__(self, time: cftime.DatetimeJulian, state: State,) -> Diagnostics:
        """Execute the radiation computations

        Args:
            time: Current model time
            state: Mapping of variable names to model state data arrays
        
        """
        self._set_compute_flags(time)
        if self._gfs_physics_control.lslwr or self._gfs_physics_control.lsswr:
            self._cached = self._compute_radiation(time, state)
        return self._cached

    def _compute_radiation(
        self, time: cftime.DatetimeJulian, state: State
    ) -> Diagnostics:
        self._rad_update(time, self._timestep)
        return self._rad_compute(state, time)

    def _set_compute_flags(self, time: cftime.DatetimeJulian) -> None:
        timestep_index = _get_forecast_time_index(time, self._init_time, self._timestep)
        self._gfs_physics_control.lsswr = _is_compute_timestep(
            timestep_index, self._gfs_physics_control.nsswr
        )
        self._gfs_physics_control.lslwr = _is_compute_timestep(
            timestep_index, self._gfs_physics_control.nslwr
        )

    def _rad_update(self, time: cftime.DatetimeJulian, dt_atmos: float) -> None:
        """Update the radiation driver's time-varying parameters"""
        if self._driver is None:
            raise ValueError(
                "Radiation driver is not set. Call `Radiation.init_driver` first."
            )
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
        fhswr = np.array(float(self._gfs_physics_control.fhswr))
        dt_atmos = np.array(float(dt_atmos))
        self._driver.radupdate(
            idat,
            jdat,
            fhswr,
            dt_atmos,
            self._gfs_physics_control.lsswr,
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
        if self._driver is None:
            raise ValueError(
                "Radiation driver is not set. Call `Radiation.init_driver` first."
            )
        solhr = self._solar_hour(time)
        statein = get_statein(state, self._tracer_inds, self._rad_config.ivflip)
        grid, coords = get_grid(state)
        sfcprop = get_sfcprop(state)
        ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
        random_numbers = io.generate_random_numbers(ncolumns, nz, NGPTSW, NGPTLW)
        out = self._driver._GFS_radiation_driver(
            self._gfs_physics_control,
            self._driver.solar_constant,
            solhr,
            statein,
            sfcprop,
            grid,
            random_numbers,
            self._lw_lookup,
            self._sw_lookup,
        )
        return unstack(postprocess_out(out), coords)

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
