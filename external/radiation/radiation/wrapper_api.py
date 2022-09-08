from typing import (
    MutableMapping,
    Mapping,
    Any,
    Hashable,
)
from mpi4py import MPI
import cftime
import numpy as np
import xarray as xr
import radiation
from radiation import io, preprocessing


State = MutableMapping[Hashable, xr.DataArray]
Diagnostics = MutableMapping[Hashable, xr.DataArray]


class Radiation:
    def __init__(
        self,
        rad_config: MutableMapping[Hashable, Any],
        comm: MPI.COMM_WORLD,
        timestep: float,
        tracer_inds: Mapping[str, int],
    ):
        self._driver: radiation.RadiationDriver = radiation.RadiationDriver()
        self._rad_config: MutableMapping[Hashable, Any] = rad_config
        self._comm: MPI.COMM_WORLD = comm
        self._timestep: float = timestep
        self._tracer_inds: Mapping[str, int] = tracer_inds

        self._download_radiation_assets()
        self._init_driver()

    def _download_radiation_assets(
        self,
        lookup_data_path: str = radiation.LOOKUP_DATA_PATH,
        forcing_data_path: str = radiation.FORCING_DATA_PATH,
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
        aerosol_data = io.load_aerosol(self._forcing_local_dir)
        sfc_filename, sfc_data = io.load_sfc(self._forcing_local_dir)
        solar_filename, _ = io.load_astronomy(
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

    # this will end up in the radiation package
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
        fhswr = np.array(float(self._rad_config["fhswr"]))
        dt_atmos = np.array(float(dt_atmos))
        aerosol_data = io.load_aerosol(self._forcing_local_dir)
        _, solar_data = io.load_astronomy(
            self._forcing_local_dir, self._rad_config["isolar"]
        )
        gas_data = io.load_gases(self._forcing_local_dir, self._rad_config["ictmflg"])
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

    def _rad_compute(self, state: State, time: cftime.DatetimeJulian,) -> Diagnostics:
        """Compute the radiative fluxes"""
        statein = preprocessing.statein(
            state, self._tracer_inds, self._rad_config["ivflip"]
        )
        grid, coords = preprocessing.grid(state)
        sfcprop = preprocessing.sfcprop(state)
        ncolumns, nz = statein["tgrs"].shape[0], statein["tgrs"].shape[1]
        model = preprocessing.model(
            self._rad_config,
            self._tracer_inds,
            time,
            self._timestep,
            nz,
            self._comm.rank,
        )
        random_numbers = io.generate_random_numbers(
            ncolumns, nz, radiation.NGPTSW, radiation.NGPTLW
        )
        lw_lookup = io.load_lw(self._lookup_local_dir)
        sw_lookup = io.load_sw(self._lookup_local_dir)
        out = self._driver._GFS_radiation_driver(
            model, statein, sfcprop, grid, random_numbers, lw_lookup, sw_lookup
        )
        out = preprocessing.postprocess_out(out)
        return preprocessing.unstack(out, coords)
