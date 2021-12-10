import xarray as xr
import numpy as np


def _center_to_interface(f: np.ndarray) -> np.ndarray:
    """Interpolate vertically cell centered data to the interface
    with linearly extrapolated inputs"""
    f_low = 2 * f[..., 0] - f[..., 1]
    f_high = 2 * f[..., -1] - f[..., -2]
    pad = np.concatenate([f_low[..., np.newaxis], f, f_high[..., np.newaxis]], axis=-1)
    return (pad[..., :-1] + pad[..., 1:]) / 2


def _convergence(eddy: np.ndarray, delp: np.ndarray) -> np.ndarray:
    padded = _center_to_interface(eddy)
    # pad interfaces assuming eddy = 0 at edges
    return -np.diff(padded, axis=-1) / delp


def cell_center_convergence(
    eddy: xr.DataArray, delp: xr.DataArray, dim: str = "p"
) -> xr.DataArray:
    """Compute vertical convergence of a cell-centered flux.

    This flux is assumed to vanish at the vertical boundaries
    """
    return xr.apply_ufunc(
        _convergence,
        eddy,
        delp,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[eddy.dtype],
    )


def project_vertical_flux(
    field: xr.DataArray,
    pressure_thickness: xr.DataArray,
    first_level_flux: xr.DataArray,
    last_level_flux: xr.DataArray,
    vertical_dim: str = "z",
) -> xr.DataArray:
    """
    Produce a flux field that best represents the provided field using least squares.

    Args:
        field: field to be represented by vertical flux, should have mass-normalized
            units such as kg/kg or energy/kg
        pressure_thickness: thickness of each layer in Pascals
        first_level_flux: flux of the level at z=0, in the positive index direction
        last_level_flux: flux of the level at z=-1, in the positive index direction
        vertical_dim: name of vertical dimension
    
    Returns:
        flux_field: Flux of field * mass * g in units of
            (<field units> * kg * g)/m^2/s,
            where g is gravitational acceleration, with its vertical
            dimension labelled as "z_interface", and its first and last values
            along that axis equal to first_level_flux and last_level_flux.
    """
    # g is folded into the flux for convenient computation, since we otherwise
    # divide by g when converting to flux and multiply by g when converting back
    # to tendency fields
    return xr.apply_ufunc(
        _project_vertical_flux,
        field,
        pressure_thickness,
        first_level_flux,
        last_level_flux,
        input_core_dims=[[vertical_dim], [vertical_dim], [], []],
        output_core_dims=[["z_interface"]],
        dask="parallelized",
        output_dtypes=field.dtype,
    )


def _project_vertical_flux(
    field: xr.DataArray,
    pressure_thickness: xr.DataArray,
    first_level_flux: xr.DataArray,
    last_level_flux: xr.DataArray,
) -> xr.DataArray:
    nz = field.shape[-1]
    field = field * pressure_thickness
    field[..., 0] -= first_level_flux
    field[..., -1] += last_level_flux
    M = np.zeros((nz, nz - 1))
    M[0, 0] = -1
    M[-1, -1] = 1
    for i in range(1, nz - 1):
        M[i, i] = -1
        M[i, i - 1] = 1
    columns = int(np.product(field.shape) / nz)
    b = field.T.reshape(nz, columns)
    lstsq_result = np.linalg.lstsq(M, b)
    middle_flux_data = lstsq_result[0].reshape((nz - 1,) + tuple(field.T.shape[1:])).T
    flux_data = np.concatenate(
        [first_level_flux[..., None], middle_flux_data, last_level_flux[..., None]],
        axis=-1,
    )
    return flux_data


def flux_convergence(
    flux: xr.DataArray,
    pressure_thickness: xr.DataArray,
    vertical_dim: str = "z",
    vertical_interface_dim: str = "z_interface",
) -> xr.DataArray:
    """
    Reconstruct a field from a flux.

    Args:
        flux: Flux of field * mass * g in units of (<field units> * kg * g)/m^2/s,
            where g is gravitational acceleration, with its vertical
            dimension labelled as "z_interface", and its first and last values
            along that axis equal to first_level_flux and last_level_flux,
            defined on cell interfaces
        pressure_thickness: thickness of each layer in Pascals, defined on cell
            centers
        vertical_dim: name of the cell-center vertical dimension
        vertical_interface_dim: name of the cell-interface vertical dimension
    
    Returns:
        field: field reconstructed from the given vertical flux
    """
    return xr.apply_ufunc(
        _flux_convergence,
        flux,
        pressure_thickness,
        input_core_dims=[[vertical_interface_dim], [vertical_dim]],
        output_core_dims=[[vertical_dim]],
        dask="parallelized",
        output_dtypes=flux.dtype,
    )


def _flux_convergence(
    flux: xr.DataArray, pressure_thickness: xr.DataArray,
) -> xr.DataArray:
    return (flux[..., :-1] - flux[..., 1:]) / pressure_thickness
