import pytest
import xarray as xr

from runtime.limiters import limit_sphum_tendency_for_moisture_conservation

total_precip_zero = xr.DataArray(data=[0.0, 0.0, 0.0], dims=["x"])
total_precip_some_nonzero = xr.DataArray(data=[1.0, 3.0, 0.0], dims=["x"])
sphum_tendency_all_columns_negative = xr.DataArray(
    data=[[-0.1, -0.1], [-0.2, -0.2], [-0.2, -0.2]], dims=["x", "z"]
)
sphum_tendency_mixed_columns = xr.DataArray(
    data=[[0.1, 0.1], [0.1, 0.1], [-0.2, -0.2]], dims=["x", "z"]
)
sphum_tendency_mixed_within_columns = xr.DataArray(
    data=[[-0.1, 0.1], [-0.2, 0.1], [-0.2, 0.3]], dims=["x", "z"]
)
DELP = xr.DataArray(
    data=[[9.81, 9.81], [9.81, 9.81], [9.81, 9.81]], dims=["x", "z"]
)  # make delp/g weights all 1 for simplicity
DT = 10
m_per_mm = 1 / 1000


@pytest.mark.parametrize(
    ["total_precip", "sphum_tendency", "dt", "expected"],
    [
        pytest.param(
            m_per_mm * total_precip_zero,  # type: ignore
            sphum_tendency_all_columns_negative,
            10.0,
            sphum_tendency_all_columns_negative,
            id="zero_precip_negative_tendency",
        ),
        pytest.param(
            m_per_mm * total_precip_zero,  # type: ignore
            sphum_tendency_mixed_columns,
            10.0,
            xr.DataArray(data=[[0.0, 0.0], [0.0, 0.0], [-0.2, -0.2]], dims=["x", "z"]),
            id="zero_precip_mixed_columns",
        ),
        pytest.param(
            m_per_mm * total_precip_some_nonzero,  # type: ignore
            sphum_tendency_mixed_columns,
            10.0,
            xr.DataArray(data=[[0.0, 0.0], [0.1, 0.1], [-0.2, -0.2]], dims=["x", "z"]),
            id="nonzero_precip_mixed_columns",
        ),
        pytest.param(
            m_per_mm * total_precip_some_nonzero,  # type: ignore
            sphum_tendency_mixed_columns,
            20.0,
            xr.DataArray(data=[[0.0, 0.0], [0.0, 0.0], [-0.2, -0.2]], dims=["x", "z"]),
            id="nonzero_precip_mixed_columns_dt_20",
        ),
        pytest.param(
            m_per_mm * total_precip_zero,  # type: ignore
            sphum_tendency_mixed_within_columns,
            10.0,
            xr.DataArray(data=[[-0.1, 0.1], [-0.2, 0.1], [0.0, 0.0]], dims=["x", "z"]),
            id="mixed_within_columns",
        ),
    ],
)
def test_limit_sphum_tendency_for_moisture_conservation(
    total_precip, sphum_tendency, dt, expected
):
    updated_tendency = limit_sphum_tendency_for_moisture_conservation(
        total_precip, sphum_tendency, DELP, dt
    )
    xr.testing.assert_allclose(updated_tendency, expected)
