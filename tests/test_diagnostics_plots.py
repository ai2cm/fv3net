import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import pytest
import xarray as xr

from fv3net.diagnostics.utils import PlotConfig
from fv3net.diagnostics.visualize import (
    plot_time_series
)

@pytest.fixture()
def test_ds_time_series():
    coords = {"initialization_time": range(100) }
    da_diag_var = xr.DataArray(np.linspace(-50, 50, 100)**2, dims=["initialization_time"], coords=coords)
    ds = xr.Dataset({"diag_var": da_diag_var})
    return ds



@pytest.mark.mpl_image_compare(filename='time_series.png')
def test_plot_time_series(test_ds_time_series):
    plot_config = PlotConfig(
        diagnostic_variable='diag_var',
        plot_name='test time series',
        plotting_function='plot_time_series',
        dim_slices={},
        functions=[],
        function_kwargs=[],
        plot_kwargs={'xlabel': 'time [d]', 'ylabel': 'diag var'}
    )
    fig = plot_time_series(test_ds_time_series, plot_config)
    return fig


@pytest.mark.mpl_image_compare(filename='single_map.png')
def test_plot_diag_var_map(test):
