import vcm
from vcm.calc.thermo import _GRAVITY
import numpy as np
import xarray as xr
import pytest


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_project_conservative_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    q = _GRAVITY / delp * (flux_array[:, :, :-1] - flux_array[:, :, 1:])
    result = vcm.project_vertical_flux(
        field=xr.DataArray(q, dims=dims_center,),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        first_level_flux=xr.DataArray(flux_array[:, :, 0], dims=dims_base),
        last_level_flux=xr.DataArray(flux_array[:, :, -1], dims=dims_base),
        vertical_dim=vertical_dim,
    )
    np.testing.assert_almost_equal(result.values, flux_array)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_project_arbitrary_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    q = np.random.uniform(low=-2, high=2, size=tuple(base_shape) + (nz,))
    first_level_flux = xr.DataArray(np.random.randn(*base_shape), dims=dims_base)
    last_level_flux = xr.DataArray(np.random.randn(*base_shape), dims=dims_base)
    result = vcm.project_vertical_flux(
        field=xr.DataArray(q, dims=dims_center),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        first_level_flux=first_level_flux,
        last_level_flux=last_level_flux,
        vertical_dim=vertical_dim,
    )
    np.testing.assert_almost_equal(result.values[:, :, 0], first_level_flux.values)
    np.testing.assert_almost_equal(result.values[:, :, -1], last_level_flux.values)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_project_nonconservative_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    q_base = _GRAVITY / delp * (flux_array[:, :, :-1] - flux_array[:, :, 1:])
    q_source = np.random.uniform(low=0, high=0.1, size=q_base.shape)
    result = vcm.project_vertical_flux(
        field=xr.DataArray(q_base + q_source, dims=dims_center),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        first_level_flux=xr.DataArray(flux_array[:, :, 0], dims=dims_base),
        last_level_flux=xr.DataArray(flux_array[:, :, -1], dims=dims_base),
        vertical_dim=vertical_dim,
    )
    np.testing.assert_almost_equal(result.values[:, :, 0], flux_array[:, :, 0])
    np.testing.assert_almost_equal(result.values[:, :, -1], flux_array[:, :, -1])


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
@pytest.mark.parametrize(
    "vertical_interface_dim", ["z", "another_vertical_interface_dim"]
)
def test_construct_from_flux(vertical_dim: str, vertical_interface_dim: str):
    # this mostly tests the DataArray containerization of the results,
    # the data test itself is only that the numpy implementation here gives
    # the same answer as the implementation in the routine
    nz = 5
    base_shape = (2, 10)
    flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = dims_base + [vertical_dim]
    dims_interface = dims_base + [vertical_interface_dim]
    q = _GRAVITY / delp * (flux_array[:, :, :-1] - flux_array[:, :, 1:])
    result = vcm.flux_convergence(
        flux=xr.DataArray(flux_array, dims=dims_interface),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        vertical_dim=vertical_dim,
        vertical_interface_dim=vertical_interface_dim,
    )
    np.testing.assert_almost_equal(result.values, q)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_project_and_reconstruct_nonconservative_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    nominal_flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    pressure_thickness = xr.DataArray(delp, dims=dims_center)
    q_base = (
        _GRAVITY / delp * (nominal_flux_array[:, :, :-1] - nominal_flux_array[:, :, 1:])
    )
    q_source = np.random.uniform(low=0, high=0.1, size=q_base.shape)
    q_target = q_base + q_source
    flux_field = vcm.project_vertical_flux(
        field=xr.DataArray(q_target, dims=dims_center,),
        pressure_thickness=pressure_thickness,
        first_level_flux=xr.DataArray(nominal_flux_array[:, :, 0], dims=dims_base),
        last_level_flux=xr.DataArray(nominal_flux_array[:, :, -1], dims=dims_base),
        vertical_dim=vertical_dim,
    )
    reconstructed = vcm.flux_convergence(
        flux=flux_field,
        pressure_thickness=pressure_thickness,
        vertical_dim=vertical_dim,
    )
    rmse = np.std(reconstructed.values - q_target)
    q_source_std = np.std(q_source)
    assert rmse < q_source_std
