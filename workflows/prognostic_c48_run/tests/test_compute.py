import numpy as np
import xarray as xr

from runtime.diagnostics.compute import precipitation_sum, KG_PER_M2_PER_M


def test_precipitation_sum():
    dt = 10.0
    da = xr.DataArray(np.ones(3), dims=["x"],)
    physics_precip = da
    postphysics_integrated_moistening = -2 * da  # m
    dq2 = postphysics_integrated_moistening * KG_PER_M2_PER_M / dt  # kg/m^2/s
    sum = precipitation_sum(physics_precip, dq2, dt)
    np.testing.assert_array_equal(
        sum, physics_precip - postphysics_integrated_moistening
    )


def test_precipitation_sum_empty_dq2():
    dt = 10
    da = xr.DataArray(np.ones(3), dims=["x"],)
    physics_precip = da
    # empty data array returned when steppers do not have net moistening diagnostic
    dq2 = xr.DataArray()
    sum = precipitation_sum(physics_precip, dq2, dt)
    np.testing.assert_array_equal(sum, physics_precip)
