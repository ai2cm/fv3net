import xarray as xr
from fv3net.diagnostics.offline.compute_metrics import merge_metrics


def test_merge_metrics():
    metrics = [
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
    output = merge_metrics(metrics)
    assert set(output.data_vars) == {"a_mse", "b_mse", "a_variance", "b_variance"}
