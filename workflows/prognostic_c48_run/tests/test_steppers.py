import pytest
import xarray as xr
from runtime.steppers.machine_learning import non_negative_sphum


sphum = 1.0e-3 * xr.DataArray(data=[1.0, 1.0, 1.0], dims=["x"])  # type: ignore
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
        pytest.param(sphum, dQ1, dQ2, timestep, dQ1, dQ2, id="no_limiting"),
        #         pytest.param(
        #         sphum, dQ1, 2.0 * dQ2, timestep, dQ1 / 2.0, dQ2, id="dQ2_2x_too_big"
        #         ),
        # pytest.param(sphum, dQ1, dQ2_mixed, timestep, dQ1_reduced, dQ2, id="dQ2_mixed"
        #         pytest.param(
        #     sphum, dQ1, dQ2, 2.0 * timestep, dQ1 / 2.0, dQ2 / 2.0, id="timestep_2x"
        #         ),
    ],
)
def test_non_negative_sphum(sphum, dQ1, dQ2, dt, dQ1_expected, dQ2_expected):
    dQ1_updated, dQ2_updated = non_negative_sphum(sphum, dQ1, dQ2, dt)
    xr.testing.assert_allclose(dQ1_updated, dQ1_expected)
    xr.testing.assert_allclose(dQ2_updated, dQ2_expected)
