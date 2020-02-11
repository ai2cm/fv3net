from metpy.interpolate import interpolate_1d
import numpy as np
import xarray as xr

from ..calc.thermo import pressure_at_interface, pressure_at_midpoint_log
from ..cubedsphere import edge_weighted_block_average, weighted_block_average
from ..cubedsphere.coarsen import block_upsample_like
from ..cubedsphere.constants import (
    RESTART_Z_CENTER,
    RESTART_Z_OUTER,
    FV_CORE_X_CENTER,
    FV_CORE_X_OUTER,
    FV_CORE_Y_CENTER,
    FV_CORE_Y_OUTER,
    PRESSURE_GRID,
)
from ..regrid import regrid_to_shared_coords
from .xgcm import create_fv3_grid

try:
    import mappm
except ImportError:
    _mappm_installed = False
else:
    _mappm_installed = True


def regrid_to_common_pressure(da_var, delp):
    """ Convenience function that uses regrid_to_shared_coords() for a common
    usage of interpolating to a pressure grid

    Args:
        da_var: data array variable to regrid
        delp: data array with delp (pressure thickness)

    Returns:
        data array of da_var at vertical coordinates of
        pressure grid from vcm.cubedsphere.constants
    """
    return regrid_to_shared_coords(
        da_var,
        np.array(PRESSURE_GRID),
        pressure_at_midpoint_log(delp),
        regrid_dim_name="pressure",
        replace_dim_name="pfull",
    )


def regrid_to_area_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    area: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_CENTER,
    z_dim: str = RESTART_Z_CENTER,
) -> (xr.Dataset, xr.DataArray):
    """ Vertically regrid a dataset of cell-centered quantities to coarsened
    pressure levels.

    Args:
        ds: input Dataset
        delp: pressure thicknesses
        area: area weights
        coarsening_factor: coarsening-factor for pressure levels
        x_dim (optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (optional): y-dimension name. Defaults to "yaxis_2"
        z_dim (optional): z-dimension name. Defaults to "zaxis_1"

    Returns:
        tuple of regridded input Dataset and area masked wherever coarse
        pressure bottom interfaces are below fine surface pressure
    """
    delp_coarse = weighted_block_average(
        delp, area, coarsening_factor, x_dim=x_dim, y_dim=y_dim
    )
    return _regrid_given_delp(
        ds, delp, delp_coarse, area, x_dim=x_dim, y_dim=y_dim, z_dim=z_dim
    )


def regrid_to_edge_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    length: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_OUTER,
    z_dim: str = RESTART_Z_CENTER,
    edge: str = "x",
) -> (xr.Dataset, xr.DataArray):
    """ Vertically regrid a dataset of edge-valued quantities to coarsened
    pressure levels.

    Args:
        ds: input Dataset
        delp: pressure thicknesses
        length: edge length weights
        coarsening_factor: coarsening-factor for pressure levels
        x_dim (optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (optional): y-dimension name. Defaults to "yaxis_1"
        z_dim (optional): z-dimension name. Defaults to "zaxis_1"
        edge (optional): grid cell side to coarse-grain along {"x", "y"}

    Returns:
        tuple of regridded input Dataset and length masked wherever coarse
        pressure bottom interfaces are below fine surface pressure
    """
    hor_dims = {"x": x_dim, "y": y_dim}
    grid = create_fv3_grid(
        xr.Dataset({"delp": delp}),
        x_center=FV_CORE_X_CENTER,
        x_outer=FV_CORE_X_OUTER,
        y_center=FV_CORE_Y_CENTER,
        y_outer=FV_CORE_Y_OUTER,
    )
    interp_dim = "x" if edge == "y" else "y"
    delp_staggered = grid.interp(delp, interp_dim).assign_coords(
        {hor_dims[interp_dim]: np.arange(1, delp.sizes[hor_dims[edge]] + 2)}
    )
    delp_staggered_coarse = edge_weighted_block_average(
        delp_staggered, length, coarsening_factor, x_dim=x_dim, y_dim=y_dim, edge=edge
    )
    return _regrid_given_delp(
        ds,
        delp_staggered,
        delp_staggered_coarse,
        length,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )


def _regrid_given_delp(
    ds,
    delp_fine,
    delp_coarse,
    weights,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_CENTER,
    z_dim: str = RESTART_Z_CENTER,
):
    """Given a fine and coarse delp, do vertical regridding to coarse pressure levels
    and mask weights below fine surface pressure.
    """
    delp_coarse_on_fine = block_upsample_like(
        delp_coarse, delp_fine, x_dim=x_dim, y_dim=y_dim
    )
    phalf_coarse_on_fine = pressure_at_interface(
        delp_coarse_on_fine, dim_center=z_dim, dim_outer=RESTART_Z_OUTER
    )
    phalf_fine = pressure_at_interface(
        delp_fine, dim_center=z_dim, dim_outer=RESTART_Z_OUTER
    )

    ds_regrid = xr.zeros_like(ds)
    for var in ds:
        ds_regrid[var] = regrid_vertical(
            phalf_fine, ds[var], phalf_coarse_on_fine, z_dim_center=z_dim
        )

    masked_weights = _mask_weights(
        weights, phalf_coarse_on_fine, phalf_fine, dim_center=z_dim
    )

    return ds_regrid, masked_weights


def _mask_weights(
    weights,
    phalf_coarse_on_fine,
    phalf_fine,
    dim_center=RESTART_Z_CENTER,
    dim_outer=RESTART_Z_OUTER,
):
    return weights.where(
        phalf_coarse_on_fine.isel({dim_outer: slice(1, None)}).variable
        < phalf_fine.isel({dim_outer: -1}).variable,
        other=0.0,
    ).rename({dim_outer: dim_center})


def regrid_vertical(
    p_in: xr.DataArray,
    f_in: xr.DataArray,
    p_out: xr.DataArray,
    iv: int = 1,
    kord: int = 1,
    z_dim_center: str = RESTART_Z_CENTER,
    z_dim_outer: str = RESTART_Z_OUTER,
) -> xr.DataArray:
    """Do vertical regridding using Fortran mappm subroutine.

    Args:
        p_in: pressure at layer edges in original vertical coordinate
        f_in: variable to be regridded, defined for layer averages
        p_out: pressure at layer edges in new vertical coordinate
        iv (optional): flag for monotinicity conservation method. Defaults to 1.
            comments from mappm indicate that iv should be chosen depending on variable:
            iv = -2: vertical velocity
            iv = -1: winds
            iv = 0: positive definite scalars
            iv = 1: others
            iv = 2: temperature
        kord (optional): method number for vertical regridding. Defaults to 1.
        z_dim_center (optional): name of centered z-dimension. Defaults to "zaxis_1".
        z_dim_outer (optional): name of staggered z-dimension. Defaults to "zaxis_2".

    Returns:
        f_in regridded to p_out pressure levels

    Raises:
        ImportError: if mappm is not installed. Try `pip install vcm/external/mappm`.
    """
    if not _mappm_installed:
        raise ImportError(
            "mappm must be installed to use regrid_vertical. "
            "Try `pip install vcm/external/mappm`. Requires a Fortran compiler."
        )
    f_in_dims = f_in.dims
    dims_except_z = f_in.isel({z_dim_center: 0}).dims
    # ensure dims are in same order for all inputs
    p_in = p_in.transpose(*dims_except_z, z_dim_outer)
    p_out = p_out.transpose(*dims_except_z, z_dim_outer)
    f_in = f_in.transpose(*dims_except_z, z_dim_center)
    # reshape to 2D array for mappm
    p_in = p_in.stack(column=dims_except_z).transpose("column", z_dim_outer)
    p_out = p_out.stack(column=dims_except_z).transpose("column", z_dim_outer)
    f_in = f_in.stack(column=dims_except_z).transpose("column", z_dim_center)

    n_columns = p_in.sizes["column"]
    assert (
        f_in.sizes["column"] == n_columns and p_out.sizes["column"] == n_columns
    ), "All dimensions except vertical must be same size for p_in, f_in and p_out"
    assert (
        f_in.sizes[z_dim_center] == p_in.sizes[z_dim_outer] - 1
    ), "f_in must have a vertical dimension one shorter than p_in"

    f_out = xr.zeros_like(p_out.isel({z_dim_outer: slice(0, -1)})).rename(
        {z_dim_outer: z_dim_center}
    )
    # the final argument to mappm is unused by the subroutine
    f_out.values = mappm.mappm(
        p_in.values, f_in.values, p_out.values, 1, n_columns, iv, kord, 0.0
    )
    return f_out.unstack().transpose(*f_in_dims)
