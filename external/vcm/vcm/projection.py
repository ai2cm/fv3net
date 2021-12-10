import xarray as xr
import numpy as np


# TODO: move convergence from _fine_res_budget in loaders/mappers to here


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

    Eagerly loads the input data.

    Args:
        field: field to be represented by fluxes, should have mass-normalized units
            such as kg/kg or energy/kg
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
    # TODO: xr.apply_ufunc
    nz = field.sizes[vertical_dim]
    field = field * pressure_thickness
    first_field = field.isel({vertical_dim: slice(0, 1)}) - first_level_flux
    last_field = field.isel({vertical_dim: slice(nz - 1, nz)}) + last_level_flux
    middle_field = field.isel({vertical_dim: slice(1, nz - 1)})
    adjusted_field = xr.concat(
        [first_field, middle_field, last_field], dim=vertical_dim
    )
    M = np.zeros((nz, nz - 1))
    M[0, 0] = -1
    M[-1, -1] = 1
    for i in range(1, nz - 1):
        M[i, i] = -1
        M[i, i - 1] = 1
    nonvertical_axes = [dim for dim in field.dims if dim != vertical_dim]
    reordered = adjusted_field.transpose(vertical_dim, *nonvertical_axes)
    columns = int(np.product(reordered.shape) / nz)
    b = reordered.values.reshape(nz, columns)
    lstsq_result = np.linalg.lstsq(M, b)
    middle_flux_data = lstsq_result[0].reshape((nz - 1,) + tuple(reordered.shape[1:]))
    first_flux_data = first_level_flux.transpose(*nonvertical_axes).values[None, :]
    last_flux_data = last_level_flux.transpose(*nonvertical_axes).values[None, :]
    flux_data = np.concatenate(
        [first_flux_data, middle_flux_data, last_flux_data], axis=0
    )
    flux = xr.DataArray(
        flux_data, dims=["z_interface"] + list(reordered.dims[1:])
    ).transpose(*nonvertical_axes, "z_interface")
    return flux


def flux_convergence(
    flux: xr.DataArray,
    pressure_thickness: xr.DataArray,
    vertical_dim: str = "z",
    vertical_interface_dim: str = "z_interface",
) -> xr.DataArray:
    """
    Reconstruct a field from a flux.

    Eagerly loads the input data.

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
        field: field reconstructed from the given fluxes
    """
    nz = pressure_thickness.sizes[vertical_dim]
    difference_term = (
        flux.isel({vertical_interface_dim: slice(0, nz)}).values
        - flux.isel({vertical_interface_dim: slice(1, nz + 1)}).values
    )
    difference_dims = [
        dim if dim != vertical_interface_dim else vertical_dim for dim in flux.dims
    ]
    difference = xr.DataArray(difference_term, dims=difference_dims)
    return difference / pressure_thickness
