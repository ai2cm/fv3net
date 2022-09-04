from typing import MutableMapping, Mapping, Hashable, Any

LOOKUP_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/lookupdata/lookup.tar.gz"  # noqa: E501
FORCING_DATA_PATH = "gs://vcm-fv3gfs-serialized-regression-data/physics/forcing/data.tar.gz"  # noqa: 501


DEFAULT_RAD_CONFIG: Mapping[Hashable, Any] = {
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


def get_rad_config(
    physics_namelist: Mapping[Hashable, Any],
    default_rad_config: Mapping[Hashable, Any] = DEFAULT_RAD_CONFIG,
) -> MutableMapping[Hashable, Any]:
    """Generate radiation config from fv3gfs namelist to ensure
    identical. Additional values from hardcoded defaults.
    """
    rad_config: MutableMapping[Hashable, Any] = {}
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
