import xarray as xr

from fv3net.diagnostics.offline.compute_diagnostics import merge_diagnostics


def test_merge_diagnostics():
    diagnostics = [
        (
            "mse",
            {
                "a": xr.DataArray([1.0], dims=["x"]),
                "B": xr.DataArray([1.0], dims=["x"]),
            },
        ),
        (
            "variance",
            {
                "a": xr.DataArray([1.0], dims=["x"]),
                "B": xr.DataArray([1.0], dims=["x"]),
            },
        ),
    ]
    output = merge_diagnostics(diagnostics)
    assert set(output.data_vars) == {"a_mse", "b_mse", "a_variance", "b_variance"}
