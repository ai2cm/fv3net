import numpy as np
import xarray as xr
from ..cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER
from .constants import _GRAVITY

_TOA_PRESSURE = 300.0  # Pa
_REVERSE = slice(None, None, -1)


def mass_integrate(
    da: xr.DataArray, delp: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute mass-weighted vertical integral."""
    return (da * delp / _GRAVITY).sum(dim)


def pressure_at_interface(
    delp: xr.DataArray,
    toa_pressure: float = _TOA_PRESSURE,
    dim_center: str = COORD_Z_CENTER,
    dim_outer: str = COORD_Z_OUTER,
) -> xr.DataArray:
    """Compute pressure at layer interfaces.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim_center (optional): name of vertical dimension for delp.
            Defaults to "pfull".
        dim_outer (optional): name of vertical dimension for output pressure.
            Defaults to "phalf".

    Returns:
        atmospheric pressure at layer interfaces
    """
    top = xr.full_like(delp.isel({dim_center: [0]}), toa_pressure).variable
    delp_with_top = top.concat([top, delp.variable], dim=dim_center)
    pressure = delp_with_top.cumsum(dim_center)
    return _add_coords_to_interface_variable(
        pressure, delp, dim_center=dim_center
    ).rename({dim_center: dim_outer})


def height_at_interface(
    dz: xr.DataArray,
    phis: xr.DataArray,
    dim_center: str = COORD_Z_CENTER,
    dim_outer: str = COORD_Z_OUTER,
) -> xr.DataArray:
    """Compute geopotential height at layer interfaces.

    Args:
        dz: layer height thicknesses
        phis: surface geopotential
        dim_center (optional): name of vertical dimension for dz.
            Defaults to "pfull".
        dim_outer (optional): name of vertical dimension for output heights.
            Defaults to "phalf".

    Returns:
        height at layer interfaces
    """
    bottom = phis.broadcast_like(dz.isel({dim_center: [0]})) / _GRAVITY
    dzv = -dz.variable  # dz is negative in model
    dz_with_bottom = dzv.concat([dzv, bottom], dim=dim_center)
    height = (
        dz_with_bottom.isel({dim_center: _REVERSE})
        .cumsum(dim_center)
        .isel({dim_center: _REVERSE})
    )
    return _add_coords_to_interface_variable(height, dz, dim_center=dim_center).rename(
        {dim_center: dim_outer}
    )


def _add_coords_to_interface_variable(
    dv_outer: xr.Variable, da_center: xr.DataArray, dim_center: str = COORD_Z_CENTER
):
    """Assign all coords except vertical from da_center to dv_outer """
    if dim_center in da_center.coords:
        return xr.DataArray(dv_outer, coords=da_center.drop_vars(dim_center).coords)
    else:
        return xr.DataArray(dv_outer, coords=da_center.coords)


def pressure_at_midpoint(
    delp: xr.DataArray, toa_pressure: float = _TOA_PRESSURE, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute pressure at layer midpoints by linear interpolation.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(delp, toa_pressure=toa_pressure, dim_center=dim)
    return _interface_to_midpoint(pi, dim_center=dim)


def height_at_midpoint(
    dz: xr.DataArray, phis: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute geopotential height at layer midpoints by linear interpolation.

    Args:
        dz: layer height thicknesses
        phis: surface geopotential
        dim (optional): name of vertical dimension for dz. Defaults to "pfull".

    Returns:
        height at layer midpoints
    """
    zi = height_at_interface(dz, phis, dim_center=dim)
    return _interface_to_midpoint(zi, dim_center=dim)


def _interface_to_midpoint(da, dim_center=COORD_Z_CENTER, dim_outer=COORD_Z_OUTER):
    da_mid = (
        da.isel({dim_outer: slice(0, -1)}) + da.isel({dim_outer: slice(1, None)})
    ) / 2
    return da_mid.rename({dim_outer: dim_center})


def pressure_at_midpoint_log(
    delp: xr.DataArray, toa_pressure: float = _TOA_PRESSURE, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """Compute pressure at layer midpoints following Eq. 3.17 of Simmons
    and Burridge (1981), MWR.

    Args:
        delp: pressure thicknesses
        toa_pressure (optional): pressure at the top of atmosphere.
            Defaults to 300Pa.
        dim (optional): name of vertical dimension for delp. Defaults to "pfull".

    Returns:
        atmospheric pressure at layer midpoints
    """
    pi = pressure_at_interface(
        delp, toa_pressure=toa_pressure, dim_center=dim, dim_outer=dim
    )
    dlogp = np.log(pi).diff(dim)
    output = delp / dlogp
    # ensure that output chunks are the same
    # pressure at interface adds a single chunk at the model surface
    if delp.chunks:
        original_chunk_size = delp.chunks[delp.get_axis_num(dim)]
        output = output.chunk({dim: original_chunk_size})

    return output


def dz_and_top_to_phis(
    top_height: xr.DataArray, dz: xr.DataArray, dim: str = COORD_Z_CENTER
) -> xr.DataArray:
    """ Compute surface geopotential from model top height and layer thicknesses"""
    return _GRAVITY * (top_height + dz.sum(dim=dim))


def surface_pressure_from_delp(
    delp: xr.DataArray, p_toa: float = 300.0, vertical_dim: str = "z"
) -> xr.DataArray:
    """Compute surface pressure from delp in Pa
    
    Args:
        delp: DataArray of pressure layer thicknesses in Pa
        p_toa: Top of atmosphere pressure in Pa; optional, defaults to 300
        vertical_dim: Name of vertical dimension; defaults to 'z'
        
    Returns:
        surface_pressure: DataArray of surface pressure in Pa
    """

    surface_pressure = delp.sum(dim=vertical_dim) + p_toa
    surface_pressure = surface_pressure.assign_attrs(
        {"long_name": "surface pressure", "units": "Pa"}
    )

    return surface_pressure
