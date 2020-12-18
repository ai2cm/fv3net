import pytest
import xarray as xr

from runtime.limiters import (
    couple_sphum_tendency_and_surface_precip,
    limit_sphum_tendency_for_non_negativity,
)

M_PER_MM = 1 / 1000
# total_precip in m per timestep
total_precip_zero = M_PER_MM * xr.DataArray(  # type: ignore
    data=[0.0, 0.0, 0.0], dims=["x"]
)
total_precip_some_nonzero = M_PER_MM * xr.DataArray(  # type: ignore
    data=[1.0, 3.0, 0.0], dims=["x"]
)
# tendencies in kg/kg/s
sphum_tendency_all_columns_negative = xr.DataArray(
    data=[[-0.1, -0.1], [-0.2, -0.2], [-0.2, -0.2]], dims=["x", "z"]
)
sphum_tendency_mixed_sign_columns = xr.DataArray(
    data=[[0.1, 0.1], [0.1, 0.1], [-0.2, -0.2]], dims=["x", "z"]
)
sphum_tendency_mixed_sign_within_columns = xr.DataArray(
    data=[[-0.3, 0.5], [0.2, -0.1], [0.2, -0.3]], dims=["x", "z"]
)
# make delp/g weights all 1 kg/m**2 for simplicity
DELP = xr.DataArray(data=[[9.81, 9.81], [9.81, 9.81], [9.81, 9.81]], dims=["x", "z"])
# s
DT = 10.0


@pytest.mark.parametrize(
    [
        "total_precip",
        "sphum_tendency",
        "dt",
        "expected_total_precip",
        "expected_sphum_tendency",
    ],
    [
        pytest.param(
            total_precip_zero,
            sphum_tendency_all_columns_negative,
            DT,
            M_PER_MM * xr.DataArray(data=[2.0, 4.0, 4.0], dims=["x"]),  # type: ignore
            sphum_tendency_all_columns_negative,
            id="zero_precip_negative_tendency",
        ),
        pytest.param(
            total_precip_zero,
            sphum_tendency_mixed_sign_columns,
            DT,
            M_PER_MM * xr.DataArray(data=[0.0, 0.0, 4.0], dims=["x"]),  # type: ignore
            xr.DataArray(data=[[0.0, 0.0], [0.0, 0.0], [-0.2, -0.2]], dims=["x", "z"]),
            id="zero_precip_mixed_columns",
        ),
        pytest.param(
            total_precip_some_nonzero,
            sphum_tendency_mixed_sign_columns,
            DT,
            M_PER_MM * xr.DataArray(data=[0.0, 1.0, 4.0], dims=["x"]),  # type: ignore
            xr.DataArray(
                data=[[0.05, 0.05], [0.1, 0.1], [-0.2, -0.2]], dims=["x", "z"]
            ),
            id="nonzero_precip_mixed_columns",
        ),
        pytest.param(
            total_precip_some_nonzero,
            sphum_tendency_mixed_sign_columns,
            20.0,
            M_PER_MM * xr.DataArray(data=[0.0, 0.0, 8.0], dims=["x"]),  # type: ignore
            xr.DataArray(
                data=[[0.025, 0.025], [0.075, 0.075], [-0.2, -0.2]], dims=["x", "z"]
            ),
            id="nonzero_precip_mixed_columns_dt_20",
        ),
        pytest.param(
            total_precip_some_nonzero,
            sphum_tendency_mixed_sign_within_columns,
            DT,
            M_PER_MM * xr.DataArray(data=[0.0, 2.0, 1.0], dims=["x"]),  # type: ignore
            xr.DataArray(
                data=[[-0.15, 0.25], [0.2, -0.1], [0.2, -0.3]], dims=["x", "z"]
            ),
            id="mixed_within_columns",
        ),
    ],
)
def test_couple_sphum_tendency_and_surface_precip(
    total_precip, sphum_tendency, dt, expected_total_precip, expected_sphum_tendency
):
    (
        updated_total_precip,
        updated_sphum_tendency,
    ) = couple_sphum_tendency_and_surface_precip(total_precip, sphum_tendency, DELP, dt)
    xr.testing.assert_allclose(updated_total_precip, expected_total_precip)
    xr.testing.assert_allclose(updated_sphum_tendency, expected_sphum_tendency)


sphum_state = xr.DataArray(data=[[2.0, 1.0], [8.0, 4.0], [6.0, 3.0]], dims=["x", "z"])


@pytest.mark.parametrize(
    ["sphum_tendency", "sphum_state", "dt", "expected_sphum_tendency"],
    [
        pytest.param(
            sphum_tendency_all_columns_negative,
            sphum_state,
            DT,
            xr.DataArray(
                data=[[-0.1, -0.1], [-0.2, -0.2], [-0.2, -0.2]], dims=["x", "z"]
            ),
            id="no_excess_tendencies",
        ),
        pytest.param(
            sphum_tendency_all_columns_negative,
            sphum_state,
            20.0,
            xr.DataArray(
                data=[[-0.1, -0.05], [-0.2, -0.2], [-0.2, -0.15]], dims=["x", "z"]
            ),
            id="some_excess_tendencies",
        ),
    ],
)
def test_limit_sphum_tendency_for_non_negativity(
    sphum_tendency, sphum_state, dt, expected_sphum_tendency
):
    updated_sphum_tendency = limit_sphum_tendency_for_non_negativity(
        sphum_state, sphum_tendency, dt
    )
    xr.testing.assert_allclose(updated_sphum_tendency, expected_sphum_tendency)
