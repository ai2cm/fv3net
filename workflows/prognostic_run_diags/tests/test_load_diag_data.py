import pytest
import xarray as xr
import numpy as np

import fv3net.diagnostics.prognostic_run.load_diagnostic_data as load_diags


@pytest.fixture
def xr_darray():
    data = np.arange(16).reshape(4, 4)
    x = np.arange(4)
    y = np.arange(4)

    da = xr.DataArray(data, coords={"x": x, "y": y}, dims=["x", "y"],)

    return da


def test__coarsen_keeps_attrs(xr_darray):
    ds = xr.Dataset({"var1": xr_darray, "var2": xr_darray})
    ds.attrs = {"global_attr": "value"}
    ds.var1.attrs = {"units": "value"}
    output = load_diags._coarsen(ds, xr_darray, 2)
    assert ds.attrs == output.attrs
    assert ds.var1.attrs == output.var1.attrs
    assert ds.var2.attrs == output.var2.attrs


def test__get_coarsening_args(xr_darray):
    target_res = 2
    grid_entries = {4: "c4_grid_entry"}
    grid, coarsening_factor = load_diags._get_coarsening_args(
        xr_darray, target_res, grid_entries=grid_entries
    )
    assert grid == "c4_grid_entry"
    assert coarsening_factor == 2
