import numpy as np
import xarray as xr
import pytest
from fv3net.diagnostics.prognostic_run.views.movies import _MOVIE_SPECS, _non_zero

shape = (6, 12, 12)
shape_edge = (6, 13, 13)
n_center = shape[0] * shape[1] * shape[2]
n_edge = shape_edge[0] * shape_edge[1] * shape_edge[2]
dims = ("tile", "x", "y")
edge_dims = ("tile", "x_interface", "y_interface")
center_var = xr.DataArray(np.arange(n_center).reshape(shape), dims=dims)
edge_var = xr.DataArray(np.arange(n_edge).reshape(shape_edge), dims=edge_dims)
grid = xr.Dataset(
    {"lon": center_var, "lat": center_var, "lonb": edge_var, "latb": edge_var}
)


@pytest.mark.parametrize("movie_spec", _MOVIE_SPECS)
def test_plotting_functions(movie_spec, tmpdir):
    # just testing that the plotting functions don't crash
    da = xr.DataArray(np.ones(shape), dims=dims)
    ds = xr.Dataset({name: da for name in movie_spec.required_variables}).merge(grid)
    ds["time"] = "2015-01-01"
    movie_spec.plotting_function((ds, tmpdir.join(movie_spec.name + ".png")))


def test__non_zero():
    da_zeros = xr.DataArray(np.zeros(5))
    da_not_zeros = xr.DataArray(np.ones(5))
    assert not _non_zero(xr.Dataset({"a": da_zeros}), ["a"])
    assert _non_zero(xr.Dataset({"b": da_not_zeros}), ["b"])
    assert not _non_zero(xr.Dataset({"a": da_zeros, "b": da_not_zeros}), ["a"])
    assert not _non_zero(xr.Dataset({"a": da_zeros}), ["b"])
    assert _non_zero(xr.Dataset({"a": da_zeros, "b": da_not_zeros}), ["a", "b"])
