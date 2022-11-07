from radiation.config import RadiationConfig
import pytest


@pytest.mark.parametrize(
    ["namelist", "expected_rad_config"],
    [
        pytest.param({}, {"iovrsw": RadiationConfig().iovrsw}, id="default_iovrsw_1"),
        pytest.param({"iovr_sw": 0}, {"iovrsw": 0}, id="update_iovrsw"),
    ],
)
def test_radiation_config_from_physics_namelist(namelist, expected_rad_config):
    radiation_config = RadiationConfig.from_physics_namelist(namelist)
    for k, v in expected_rad_config.items():
        assert getattr(radiation_config, k) == v


@pytest.mark.parametrize(
    ["namelist", "expected_gfs_physics_control"],
    [
        pytest.param(
            {},
            {"fhswr": RadiationConfig().gfs_physics_control.fhswr},
            id="default_fhswr_3600",
        ),
        pytest.param({"fhswr": 1800.0}, {"fhswr": 1800.0}, id="update_fhswr",),
    ],
)
def test_gfs_physics_control_from_physics_namelist(
    namelist, expected_gfs_physics_control
):
    radiation_config = RadiationConfig.from_physics_namelist(namelist)
    gfs_physics_control = radiation_config.gfs_physics_control
    for k, v in expected_gfs_physics_control.items():
        assert getattr(gfs_physics_control, k) == v
