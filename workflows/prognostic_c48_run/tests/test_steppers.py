import pytest
import xarray as xr
from runtime.steppers.machine_learning import (
    non_negative_sphum,
    update_temperature_tendency_to_conserve_mse,
    update_moisture_tendency_to_ensure_non_negative_humidity,
)
import vcm


sphum = 1.0e-3 * xr.DataArray(data=[1.0, 1.0, 1.0], dims=["x"])  # type: ignore
zeros = xr.zeros_like(sphum)
dQ2 = -1.0e-5 * xr.DataArray(data=[1.0, 1.0, 1.0], dims=["x"])  # type: ignore
dQ2_mixed = -1.0e-5 * xr.DataArray(data=[1.0, 2.0, 3.0], dims=["x"])  # type: ignore
dQ1 = 1.0e-2 * xr.DataArray(data=[1.0, 1.0, 1.0], dims=["x"])  # type: ignore
dQ1_reduced = 1.0e-2 * xr.DataArray(  # type: ignore
    data=[1.0, 0.5, 1.0 / 3.0], dims=["x"]
)
timestep = 100.0


@pytest.mark.parametrize(
    ["sphum", "dQ1", "dQ2", "dt", "dQ1_expected", "dQ2_expected"],
    [
        pytest.param(
            sphum, zeros, zeros, timestep, zeros, zeros, id="all_zero_tendencies"
        ),
        pytest.param(sphum, dQ1, dQ2, timestep, dQ1, dQ2, id="no_limiting"),
        pytest.param(
            sphum, dQ1, 2.0 * dQ2, timestep, dQ1 / 2.0, dQ2, id="dQ2_2x_too_big"
        ),
        pytest.param(
            sphum, zeros, 2.0 * dQ2, timestep, zeros, dQ2, id="dQ2_2x_too_big_no_dQ1",
        ),
        pytest.param(sphum, dQ1, dQ2_mixed, timestep, dQ1_reduced, dQ2, id="dQ2_mixed"),
        pytest.param(
            sphum, dQ1, dQ2, 2.0 * timestep, dQ1 / 2.0, dQ2 / 2.0, id="timestep_2x"
        ),
    ],
)
def test_non_negative_sphum(sphum, dQ1, dQ2, dt, dQ1_expected, dQ2_expected):
    dQ1_updated, dQ2_updated = non_negative_sphum(sphum, dQ1, dQ2, dt)
    xr.testing.assert_allclose(dQ1_updated, dQ1_expected)
    xr.testing.assert_allclose(dQ2_updated, dQ2_expected)


def test_update_q2_to_ensure_non_negative_humidity():
    sphum = xr.DataArray([1, 2])
    q2 = xr.DataArray([-3, -1])
    dt = 1.0
    limited_tendency = update_moisture_tendency_to_ensure_non_negative_humidity(
        sphum, q2, dt
    )
    expected_limited_tendency = xr.DataArray([-1, -1])
    xr.testing.assert_identical(limited_tendency, expected_limited_tendency)


def test_update_q1_to_conserve_mse():
    q1 = xr.DataArray([-4, 2])
    q2 = xr.DataArray([-3, -1])
    q2_limited = xr.DataArray([-1, -1])
    q1_limited = update_temperature_tendency_to_conserve_mse(q1, q2, q2_limited)
    xr.testing.assert_identical(
        vcm.moist_static_energy_tendency(q1, q2),
        vcm.moist_static_energy_tendency(q1_limited, q2_limited),
    )
