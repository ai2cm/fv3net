from radiation.config import RadiationConfig, GFSPhysicsControlConfig
import pytest


def get_namelist(iovr_sw=None, fhswr=None, npz=None, levr=None):
    namelist = {"gfs_physics_nml": {}, "fv_core_nml": {}}
    if iovr_sw is not None:
        namelist["gfs_physics_nml"] = {"iovr_sw": iovr_sw}
    if fhswr is not None:
        namelist["gfs_physics_nml"] = {"fhswr": fhswr}
    if npz is not None:
        namelist["fv_core_nml"] = {"npz": npz}
    if levr is not None:
        namelist["gfs_physics_nml"] = {"levr": levr}
    return namelist


@pytest.mark.parametrize(
    ["namelist", "expected_rad_config"],
    [
        pytest.param(
            get_namelist(), {"iovrsw": RadiationConfig().iovrsw}, id="default_iovrsw_1"
        ),
        pytest.param(get_namelist(iovr_sw=0), {"iovrsw": 0}, id="update_iovrsw"),
    ],
)
def test_radiation_config_from_namelist(namelist, expected_rad_config):
    radiation_config = RadiationConfig.from_namelist(namelist)
    for k, v in expected_rad_config.items():
        assert getattr(radiation_config, k) == v


@pytest.mark.parametrize(
    ["namelist", "expected_gfs_physics_control_config"],
    [
        pytest.param(
            get_namelist(),
            {"fhswr": RadiationConfig().gfs_physics_control_config.fhswr},
            id="default_fhswr_3600",
        ),
        pytest.param(get_namelist(fhswr=1800.0), {"fhswr": 1800.0}, id="update_fhswr"),
        pytest.param(
            get_namelist(),
            {"levs": RadiationConfig().gfs_physics_control_config.levs},
            id="default_levs_79",
        ),
        pytest.param(get_namelist(npz=63), {"levs": 63}, id="update_levs",),
    ],
)
def test_gfs_physics_control_from_namelist(
    namelist, expected_gfs_physics_control_config
):
    gfs_physics_control_config = GFSPhysicsControlConfig.from_namelist(namelist)
    for k, v in expected_gfs_physics_control_config.items():
        assert getattr(gfs_physics_control_config, k) == v


def test_levs_same_as_levr():
    namelist = get_namelist()
    gfs_physics_control_config = GFSPhysicsControlConfig.from_namelist(namelist)
    assert gfs_physics_control_config.levs == gfs_physics_control_config.levr


def test_levr_different_error():
    namelist = get_namelist(npz=79, levr=63)
    with pytest.raises(ValueError):
        GFSPhysicsControlConfig.from_namelist(namelist)
