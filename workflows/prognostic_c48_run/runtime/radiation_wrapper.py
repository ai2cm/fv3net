import dataclasses
from typing import Optional, Mapping, Any, Literal
import cftime
import numpy as np
from runtime.steppers.machine_learning import (
    MachineLearningConfig,
    open_model,
    MultiModelAdapter,
)
from radiation import RadiationDriver, getdata


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
        """Generate radiation namelist from fv3gfs' GFS physics namelist to ensure
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
