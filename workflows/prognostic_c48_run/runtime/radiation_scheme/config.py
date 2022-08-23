import dataclasses
from typing import Optional
from runtime.steppers.machine_learning import MachineLearningConfig
import radiation
from radiation.preprocess import init_data


DEFAULT_RAD_INIT_NAMELIST = {
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
}

RAD_MODEL_NAMELIST = {
    "nfxr": 45,
    "ntrac": 8,
    "ntcw": 2,
    "ntrw": 3,
    "ntiw": 4,
    "ntsw": 5,
    "ntgl": 6,
    "ntoz": 7,
    "ntclamt": 8,
    "ncld": 5,
    "ncnd": 5,
    "fhswr": 3600,
    "fhlwr": 3600,
    "solhr": 0.0,
    "lsswr": True,
    "lslwr": True,
    "imp_physics": 11,
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
    "pertalb": [[999.0], [999.0], [999.0], [999.0], [999.0]],
    "do_only_clearsky_rad": False,
    "swhtr": True,
    "solcon": 1320.8872136343873,
    "lprnt": False,
    "lwhtr": True,
    "lssav": True,
}


@dataclasses.dataclass
class RadiationConfig:
    scheme: str
    input_model: Optional[MachineLearningConfig] = None
    offline: bool = True


def _get_rad_namelist(physics_namelist, default_rad_namelist):
    """Generate radiation namelist from fv3gfs' GFS physics namelist to ensure
    identical. Additional values from hardcoded defaults.
    """
    rad_init_namelist = {}
    rad_init_namelist["imp_physics"] = physics_namelist["imp_physics"]
    rad_init_namelist["iemsflg"] = physics_namelist["iems"]
    rad_init_namelist["ioznflg"] = default_rad_namelist["ioznflg"]
    rad_init_namelist["ictmflg"] = default_rad_namelist["ictmflg"]
    rad_init_namelist["isolar"] = physics_namelist["isol"]
    rad_init_namelist["ico2flg"] = physics_namelist["ico2"]
    rad_init_namelist["iaerflg"] = physics_namelist["iaer"]
    rad_init_namelist["ialbflg"] = physics_namelist["ialb"]
    rad_init_namelist["icldflg"] = default_rad_namelist["icldflg"]
    rad_init_namelist["ivflip"] = default_rad_namelist["ivflip"]
    rad_init_namelist["iovrsw"] = default_rad_namelist["iovrsw"]
    rad_init_namelist["iovrlw"] = default_rad_namelist["iovrlw"]
    rad_init_namelist["isubcsw"] = physics_namelist["isubc_sw"]
    rad_init_namelist["isubclw"] = physics_namelist["isubc_lw"]
    rad_init_namelist["lcrick"] = default_rad_namelist["lcrick"]
    rad_init_namelist["lcnorm"] = default_rad_namelist["lcnorm"]
    rad_init_namelist["lnoprec"] = default_rad_namelist["lnoprec"]
    rad_init_namelist["iswcliq"] = default_rad_namelist["iswcliq"]
    return rad_init_namelist


def init_radiation_driver(
    rank: int,
    physics_namelist: dict,
    forcing_dir: str = "./data/forcing",
    fv_core_dir: str = "./INPUT/",
    default_rad_init_namelist: dict = DEFAULT_RAD_INIT_NAMELIST,
) -> radiation.RadiationDriver:
    driver = radiation.RadiationDriver()
    rad_init_namelist = _get_rad_namelist(physics_namelist, default_rad_init_namelist)
    (sigma, nlay, aer_dict, solar_filename, sfc_file, sfc_data,) = init_data(
        forcing_dir, fv_core_dir, rad_init_namelist["isolar"]
    )
    return driver.radinit(
        sigma,
        nlay,
        rad_init_namelist["imp_physics"],
        rank,
        rad_init_namelist["iemsflg"],
        rad_init_namelist["ioznflg"],
        rad_init_namelist["ictmflg"],
        rad_init_namelist["isolar"],
        rad_init_namelist["ico2flg"],
        rad_init_namelist["iaerflg"],
        rad_init_namelist["ialbflg"],
        rad_init_namelist["icldflg"],
        rad_init_namelist["ivflip"],
        rad_init_namelist["iovrsw"],
        rad_init_namelist["iovrlw"],
        rad_init_namelist["isubcsw"],
        rad_init_namelist["isubclw"],
        rad_init_namelist["lcrick"],
        rad_init_namelist["lcnorm"],
        rad_init_namelist["lnoprec"],
        rad_init_namelist["iswcliq"],
        aer_dict,
        solar_filename,
        sfc_file,
        sfc_data,
    )
