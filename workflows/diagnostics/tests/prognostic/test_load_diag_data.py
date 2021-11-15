import cftime
import pytest
import xarray as xr
import numpy as np
import vcm.catalog
import joblib

import fv3net.diagnostics.prognostic_run.load_run_data as load_diags


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


def test_load_coarse_data(tmpdir):
    num_time = 2
    arr = np.arange((num_time * 6 * 48 * 48)).reshape((num_time, 6, 48, 48))
    dims = ("time", "tile", "grid_xt", "grid_yt")

    ds = xr.Dataset(
        {"a": (dims, arr)},
        coords={
            "time": [cftime.DatetimeJulian(2016, 1, n + 1) for n in range(num_time)]
        },
    )

    path = str(tmpdir.join("sfc_dt_atmos.zarr"))
    ds.to_zarr(path, consolidated=True)
    loaded = load_diags.load_coarse_data(path, vcm.catalog.catalog)
    np.testing.assert_equal(loaded["a"].values, ds.a.values)


def print_coord_hashes(ds):
    print()
    for coord in ds.coords:
        print(str(coord), joblib.hash(np.asarray(ds[coord])))


def _print_input_data_regressions_data(input_data):
    for key, (prognostic_run, verification, grid) in input_data.items():
        print(key)
        print("Prognostic Run")
        prognostic_run.info()
        print_coord_hashes(prognostic_run)

        print("Verification")
        verification.info()
        print_coord_hashes(verification)

        print("grid")
        grid.info()
        print_coord_hashes(grid)


def test_evaluation_pair_to_input_data(regtest):
    url = "gs://vcm-ml-code-testing-data/sample-prognostic-run-output"
    catalog = vcm.catalog.catalog
    prognostic = load_diags.SegmentedRun(url, catalog)
    grid = load_diags.load_grid(catalog)
    input_data = load_diags.evaluation_pair_to_input_data(prognostic, prognostic, grid)
    with regtest:
        _print_input_data_regressions_data(input_data)


@pytest.mark.parametrize(
    "simulation",
    [
        load_diags.SegmentedRun(
            url="gs://vcm-ml-code-testing-data/sample-prognostic-run-output",
            catalog=vcm.catalog.catalog,
        ),
        load_diags.CatalogSimulation("40day_may2020", vcm.catalog.catalog),
    ],
)
def test_Simulations(regtest, simulation):
    with regtest:
        simulation.data_3d.info()
        simulation.data_2d.info()
