import xarray as xr

from ..calc.thermo import pressure_at_interface
from ..cubedsphere import (
    edge_weighted_block_average,
    weighted_block_average,
)
from ..cubedsphere.coarsen import block_upsample
from ..cubedsphere.constants import (
    RESTART_Z_CENTER,
    FV_CORE_X_CENTER,
    FV_CORE_X_OUTER,
    FV_CORE_Y_CENTER,
    FV_CORE_Y_OUTER,
)
from .xgcm import create_fv3_grid

try:
    import mappm
except ImportError:
    _mappm_installed = False
else:
    _mappm_installed = True


def remap_to_area_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    area: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_CENTER,
    z_dim: str = RESTART_Z_CENTER,
):
    """ Vertically remap a dataset of cell-centered quantities to coarsened
    pressure levels.

    Args:
        ds (xr.Dataset): input Dataset
        delp (xr.DataArray): pressure thicknesses
        area (xr.DataArray): area weights
        coarsening_factor (int): coarsening-factor for pressure levels
        x_dim (str, optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (str, optional): y-dimension name. Defaults to "yaxis_2"
        z_dim (str, optional): z-dimension name. Defaults to "zaxis_1"

    Returns:
        (xr.Dataset, xr.DataArray): tuple of remapped input Dataset and area masked
        wherever coarse pressure bottom interfaces are below fine surface pressure
    """
    delp_coarse = weighted_block_average(
        delp, area, coarsening_factor, x_dim=x_dim, y_dim=y_dim
    )
    return _remap_given_delp(
        ds,
        delp,
        delp_coarse,
        area,
        coarsening_factor,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )


def remap_to_edge_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    length: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_OUTER,
    z_dim: str = RESTART_Z_CENTER,
    edge: str = "x",
):
    """ Vertically remap a dataset of edge-valued quantities to coarsened
    pressure levels.

    Args:
        ds (xr.Dataset): input Dataset
        delp (xr.DataArray): pressure thicknesses
        length (xr.DataArray): edge length weights
        coarsening_factor (int): coarsening-factor for pressure levels
        x_dim (str, optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (str, optional): y-dimension name. Defaults to "yaxis_1"
        edge (str, optional): grid cell side to coarse-grain along {"x", "y"}

    Returns:
        (xr.Dataset, xr.DataArray): tuple of remapped input Dataset and length masked
        wherever coarse pressure bottom interfaces are below fine surface pressure
    """
    grid = create_fv3_grid(
        xr.Dataset({"delp": delp}),
        x_center=FV_CORE_X_CENTER,
        x_outer=FV_CORE_X_OUTER,
        y_center=FV_CORE_Y_CENTER,
        y_outer=FV_CORE_Y_OUTER,
    )
    other_dim = "x" if edge == "y" else "y"
    delp_staggered = grid.interp(delp, other_dim)
    delp_staggered_coarse = edge_weighted_block_average(
        delp_staggered, length, coarsening_factor, x_dim=x_dim, y_dim=y_dim, edge=edge
    )
    return _remap_given_delp(
        ds,
        delp_staggered,
        delp_staggered_coarse,
        length,
        coarsening_factor,
        x_dim=x_dim,
        y_dim=y_dim,
        z_dim=z_dim,
    )


def _remap_given_delp(
    ds,
    delp_fine,
    delp_coarse,
    weights,
    coarsening_factor,
    x_dim: str = FV_CORE_X_CENTER,
    y_dim: str = FV_CORE_Y_CENTER,
    z_dim: str = RESTART_Z_CENTER,
):
    """Given a fine and coarse delp, do vertical remapping to coarse pressure levels
    and mask weights below fine surface pressure.
    """
    delp_coarse_on_fine = block_upsample(delp_coarse, coarsening_factor, [x_dim, y_dim])
    phalf_coarse_on_fine = pressure_at_interface(delp_coarse_on_fine, dim=z_dim)
    phalf_fine = pressure_at_interface(delp_fine, dim=z_dim)

    ds_remap = xr.zeros_like(ds)
    for var in ds:
        ds_remap[var] = remap_levels(
            phalf_fine, ds[var], phalf_coarse_on_fine, z_dim=z_dim
        )

    masked_weights = weights.where(
        phalf_coarse_on_fine.isel({z_dim: slice(1, None)})
        < phalf_fine.isel({z_dim: -1}),
        other=0.0,
    )

    return ds_remap, masked_weights


def remap_levels(p_in, f_in, p_out, iv=1, kord=1, z_dim=RESTART_Z_CENTER):
    """Do vertical remapping using Fortran mappm subroutine.

    Args:
        p_in (xr.DataArray): pressure at layer edges in original vertical coordinate
        f_in (xr.DataArray): variable to be remapped, defined for layer averages
        p_out (xr.DataArray): pressure at layer edges in new vertical coordinate
        iv (int, optional): flag for monotinicity conservation method. Defaults to 1.
            comments from mappm indicate that iv should be chosen depending on variable:
            iv = -2: vertical velocity
            iv = -1: winds
            iv = 0: positive definite scalars
            iv = 1: others
            iv = 2: temperature
        kord (int, optional): method number for vertical remapping. Defaults to 1.
        z_dim (str, optional): name of vertical dimension. Defaults to "zaxis_1".

    Returns:
        xr.DataArray: f_in remapped to p_out pressure levels

    Raises:
        ImportError: if mappm is not installed. Try `pip install vcm/external/mappm`.
    """
    if not _mappm_installed:
        raise ImportError(
            "mappm must be installed to use remap_levels. "
            "Try `pip install vcm/external/mappm`. Requires a Fortran compiler."
        )

    # reshape for mappm
    dims_except_z = f_in.isel({z_dim: 0}).dims
    p_in = p_in.stack(column=dims_except_z).transpose("column", z_dim)
    f_in = f_in.stack(column=dims_except_z).transpose("column", z_dim)
    p_out = p_out.stack(column=dims_except_z).transpose("column", z_dim)

    n_columns = p_in.sizes["column"]
    assert (
        f_in.sizes["column"] == n_columns and p_out.sizes["column"] == n_columns
    ), "All dimensions except vertical must be same size for p_in, f_in and p_out"
    assert (
        f_in.sizes[z_dim] == p_in.sizes[z_dim] - 1
    ), "f_in must have a vertical dimension one shorter than p_in"

    f_out = xr.zeros_like(p_out.isel({z_dim: slice(0, -1)}))
    # the final argument to mappm is unused by the subroutine
    f_out.values = mappm.mappm(
        p_in.values, f_in.values, p_out.values, 1, n_columns, iv, kord, 0.0
    )
    return f_out.unstack()
