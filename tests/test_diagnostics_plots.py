"""
If adding a new plot test, run
 > pytest tests/test_diagnostics_plots.py --mpl-generate-path tests/baseline_images
first before pytest to generate the new test image in the comparison set
"""
import numpy as np
import pytest
import xarray as xr
from vcm.calc.diag_ufuncs import mean_over_dim

from fv3net.diagnostics.utils import PlotConfig
from fv3net.diagnostics.visualize import create_plot, plot_time_series


@pytest.fixture()
def test_gridded_ds():
    centered_coords = {"tile": range(1, 7), "grid_yt": [1, 2], "grid_xt": [1, 2]}
    lont_da = xr.DataArray(
        [[[0.5, 1.5], [0.5, 1.5]] for tile in range(6)],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords,
    )
    latt_da = xr.DataArray(
        [[[0.5, 0.5], [-0.5, -0.5]] for tile in range(6)],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords,
    )
    corner_coords = {"tile": range(1, 7), "grid_y": [1, 2, 3], "grid_x": [1, 2, 3]}
    lon_grid = xr.DataArray(
        [[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]] for tile in range(6)],
        dims=["tile", "grid_y", "grid_x"],
        coords=corner_coords,
    )
    lat_grid = xr.DataArray(
        [[[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] for tile in range(6)],
        dims=["tile", "grid_y", "grid_x"],
        coords=corner_coords,
    )
    da_diag_var = xr.DataArray(
        [[[1.0, 2.0], [3.0, 4.0]] for tile in range(6)],
        dims=["tile", "grid_yt", "grid_xt"],
        coords=centered_coords,
    )
    ds = xr.Dataset(
        {
            "diag_var": da_diag_var,
            "grid_lont": lont_da,
            "grid_latt": latt_da,
            "grid_lon": lon_grid,
            "grid_lat": lat_grid,
        }
    )
    return ds


@pytest.fixture()
def test_ds_time_series():
    coords = {"pfull": [1, 2], "initialization_time": range(100)}
    da_diag_var = xr.DataArray(
        [np.linspace(50, -50, 100), np.linspace(-50, 50, 100)],
        dims=["pfull", "initialization_time"],
        coords=coords,
    )
    ds_time_series = xr.Dataset({"diag_var": da_diag_var})
    return ds_time_series


@pytest.mark.skip()
@pytest.mark.mpl_image_compare(filename="time_series_sliced.png")
def test_create_plot(test_ds_time_series):
    plot_config = PlotConfig(
        diagnostic_variable=["mean_diag_var"],
        plot_name="test time series sliced",
        plotting_function="plot_time_series",
        dim_slices={"initialization_time": slice(None, 50, None)},
        functions=[mean_over_dim],
        function_kwargs=[
            {"dim": "pfull", "var_to_avg": "diag_var", "new_var": "mean_diag_var"}
        ],
        plot_params={"xlabel": "time [d]", "ylabel": "mean diag var"},
    )
    fig = create_plot(test_ds_time_series, plot_config)
    return fig


@pytest.mark.skip()
@pytest.mark.mpl_image_compare(filename="time_series.png")
def test_plot_time_series(test_ds_time_series):
    plot_config = PlotConfig(
        diagnostic_variable=["diag_var"],
        plot_name="test time series",
        plotting_function="plot_time_series",
        dim_slices={},
        functions=[],
        function_kwargs=[],
        plot_params={"xlabel": "time [d]", "ylabel": "diag var"},
    )
    fig = plot_time_series(test_ds_time_series.isel(pfull=0), plot_config)
    return fig
