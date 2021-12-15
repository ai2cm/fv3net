import vcm
import numpy as np
import xarray as xr
import pytest


def test_convergence_cell_center_constant():
    nz = 5
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([0, 0, 0, 0, 0]).reshape((1, 1, nz))

    ans = vcm.convergence_cell_center(delp, delp)
    np.testing.assert_almost_equal(ans, expected)


def test_convergence_cell_center_linear():
    nz = 5
    f = np.arange(nz).reshape((1, 1, nz))
    delp = np.ones(nz).reshape((1, 1, nz))

    expected = np.array([-1, -1, -1, -1, -1]).reshape((1, 1, nz))

    ans = vcm.convergence_cell_center(f, delp)
    np.testing.assert_almost_equal(ans, expected)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_fit_conservative_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    q = (flux_array[:, :, :-1] - flux_array[:, :, 1:]) / delp
    result = vcm.fit_field_as_flux(
        field=xr.DataArray(q, dims=dims_center,),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        first_level_flux=xr.DataArray(flux_array[:, :, 0], dims=dims_base),
        last_level_flux=xr.DataArray(flux_array[:, :, -1], dims=dims_base),
        vertical_dim=vertical_dim,
    )
    np.testing.assert_almost_equal(result.values, flux_array)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_fit_arbitrary_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    q = np.random.uniform(low=-2, high=2, size=tuple(base_shape) + (nz,))
    first_level_flux = xr.DataArray(np.random.randn(*base_shape), dims=dims_base)
    last_level_flux = xr.DataArray(np.random.randn(*base_shape), dims=dims_base)
    result = vcm.fit_field_as_flux(
        field=xr.DataArray(q, dims=dims_center),
        pressure_thickness=xr.DataArray(delp, dims=dims_center),
        first_level_flux=first_level_flux,
        last_level_flux=last_level_flux,
        vertical_dim=vertical_dim,
    )
    np.testing.assert_almost_equal(result.values[:, :, 0], first_level_flux.values)
    np.testing.assert_almost_equal(result.values[:, :, -1], last_level_flux.values)


@pytest.mark.parametrize("vertical_dim", ["z", "another_vertical_dim"])
def test_fit_nonconservative_tendency(vertical_dim: str):
    nz = 5
    base_shape = (2, 10)
    flux_array = np.random.randn(*base_shape, nz + 1)
    delp = np.random.uniform(low=10.0, high=12.0, size=tuple(base_shape) + (nz,))
    dims_base = ["dim0", "dim1"]
    dims_center = ["dim0", "dim1", vertical_dim]
    q_base = (flux_array[:, :, :-1] - flux_array[:, :, 1:]) / delp
    q_source = np.random.uniform(low=0, high=0.1, size=q_base.shape)
    result = vcm.fit_field_as_flux(
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
    "vertical_interface_dim", ["z_interface", "another_vertical_interface_dim"]
)
def test_convergence_cell_interface(vertical_dim: str, vertical_interface_dim: str):
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
    q = (flux_array[:, :, :-1] - flux_array[:, :, 1:]) / delp
    result = vcm.convergence_cell_interface(
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
    q_base = (nominal_flux_array[:, :, :-1] - nominal_flux_array[:, :, 1:]) / delp
    q_source = np.random.uniform(low=0, high=0.1, size=q_base.shape)
    q_target = q_base + q_source
    flux_field = vcm.fit_field_as_flux(
        field=xr.DataArray(q_target, dims=dims_center,),
        pressure_thickness=pressure_thickness,
        first_level_flux=xr.DataArray(nominal_flux_array[:, :, 0], dims=dims_base),
        last_level_flux=xr.DataArray(nominal_flux_array[:, :, -1], dims=dims_base),
        vertical_dim=vertical_dim,
    )
    reconstructed = vcm.convergence_cell_interface(
        flux=flux_field,
        pressure_thickness=pressure_thickness,
        vertical_dim=vertical_dim,
    )
    rmse = np.std(reconstructed.values - q_target)
    q_source_std = np.std(q_source)
    assert rmse < q_source_std
