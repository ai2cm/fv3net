import numpy as np
import xarray as xr
import vcm
from vcm.calc.flux_form import (
    _tendency_to_flux,
    _flux_to_tendency,
    _tendency_to_implied_surface_downward_flux,
)


def get_3d_array():
    return xr.DataArray(np.random.random_sample((6, 4, 4)), dims=["z", "y", "x"])


def get_2d_array():
    return xr.DataArray(np.random.random_sample((4, 4)), dims=["y", "x"])


def test_flux_transform_round_trip():
    tendency = get_3d_array()
    toa_net_flux = get_2d_array()
    surface_upward_flux = get_2d_array()
    delp = get_3d_array()
    net_flux_column, surface_downward_flux = _tendency_to_flux(
        tendency, toa_net_flux, surface_upward_flux, delp, dim="z", rectify=False
    )
    round_tripped_tendency = _flux_to_tendency(
        net_flux_column, surface_downward_flux, surface_upward_flux, delp, dim="z"
    )
    xr.testing.assert_allclose(tendency, round_tripped_tendency)


def test_implied_surface_downward_flux():
    tendency = get_3d_array()
    toa_net_flux = get_2d_array()
    surface_upward_flux = get_2d_array()
    delp = get_3d_array()
    surface_downward_flux = _tendency_to_implied_surface_downward_flux(
        tendency, toa_net_flux, surface_upward_flux, delp, dim="z", rectify=False
    )
    xr.testing.assert_allclose(
        vcm.mass_integrate(tendency, delp, "z"),
        toa_net_flux + surface_upward_flux - surface_downward_flux,
    )
