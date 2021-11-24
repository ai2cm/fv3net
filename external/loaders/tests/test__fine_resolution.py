import numpy as np
import pytest
import xarray as xr
from loaders.mappers._fine_res import (
    Approach,
    _extend_lower,
    compute_budget,
)


@pytest.mark.parametrize(
    ["nx", "nz", "vertical_dim", "n_levels", "error"],
    (
        pytest.param(1, 5, "z", 2, False, id="base"),
        pytest.param(5, 1, "x", 2, False, id="different_dim"),
        pytest.param(1, 5, "z", 3, False, id="different_n_levels"),
        pytest.param(5, 1, "z", 2, True, id="scalar_dim"),
    ),
)
def test__extend_lower(nz, nx, vertical_dim, n_levels, error):
    f = xr.DataArray(np.arange(nx * nz).reshape((nx, nz)), dims=["x", "z"])
    f.attrs = {"long_name": "some source term", "description": "some source term"}
    ds = xr.Dataset({"f": f})
    if error:
        with pytest.raises(ValueError):
            extended = _extend_lower(ds.f, vertical_dim, n_levels)
    else:
        extended = _extend_lower(ds.f, vertical_dim, n_levels)
        expected = extended.isel({vertical_dim: -(n_levels + 1)})
        [
            xr.testing.assert_allclose(
                extended.isel({vertical_dim: -(i + 1)}), expected
            )
            for i in range(n_levels)
        ]


@pytest.mark.parametrize("include_temperature_nudging", [True, False])
@pytest.mark.parametrize("approach", list(Approach))
def test_compute_budget(approach, include_temperature_nudging):
    one = xr.DataArray(np.ones((5, 1, 1)), dims=["z", "y", "x"])
    ds = xr.Dataset()
    for name in [
        "delp",
        "T",
        "dq3dt_deep_conv_coarse",
        "dq3dt_mp_coarse",
        "dq3dt_pbl_coarse",
        "dq3dt_shal_conv_coarse",
        "dt3dt_deep_conv_coarse",
        "dt3dt_lw_coarse",
        "dt3dt_mp_coarse",
        "dt3dt_ogwd_coarse",
        "dt3dt_pbl_coarse",
        "dt3dt_shal_conv_coarse",
        "dt3dt_sw_coarse",
        "eddy_flux_vulcan_omega_sphum",
        "eddy_flux_vulcan_omega_temp",
        "exposed_area",
        "qv_dt_fv_sat_adj_coarse",
        "qv_dt_phys_coarse",
        "sphum",
        "sphum_storage",
        "sphum_vulcan_omega_coarse",
        "t_dt_fv_sat_adj_coarse",
        "t_dt_nudge_coarse",
        "t_dt_phys_coarse",
        "vulcan_omega_coarse",
        "T_vulcan_omega_coarse",
        "T_storage",
        "air_temperature_tendency_due_to_nudging",
        "specific_humidity_tendency_due_to_nudging",
        "tendency_of_air_temperature_due_to_dynamics",
        "tendency_of_specific_humidity_due_to_dynamics",
    ]:
        ds[name] = one

    out = compute_budget(ds, approach, include_temperature_nudging)
    assert {"dQ1", "dQ2"} <= set(out)
