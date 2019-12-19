import xarray as xr

from ..calc import thermo
from ..cubedsphere import (
    edge_weighted_block_average,
    weighted_block_average,
)
from ..cubedsphere.coarsen import block_upsample
from .xgcm import create_fv3_grid

try:
    import mappm
except ImportError:
    _mappm_installed = False
else:
    _mappm_installed = True

# default for restart file
VERTICAL_DIM = "zaxis_1"


def remap_to_area_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    area: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = "xaxis_1",
    y_dim: str = "yaxis_1",
):
    """ Vertically remap a dataset of cell-centered quantities to coarsened pressure levels.

    Args:
        ds (xr.Dataset): input Dataset
        delp (xr.DataArray): pressure thicknesses
        area (xr.DataArray): area weights
        coarsening_factor (int): coarsening-factor for pressure levels
        x_dim (str, optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (str, optional): y-dimension name. Defaults to "yaxis_1"

    Returns:
        xr.Dataset
    """
    delp_coarse = weighted_block_average(
        delp, area, coarsening_factor, x_dim=x_dim, y_dim=y_dim
    )
    return _remap_given_delp(
        ds, delp, delp_coarse, area, coarsening_factor, x_dim=x_dim, y_dim=y_dim
    )


def remap_to_edge_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    length: xr.DataArray,
    coarsening_factor: int,
    x_dim: str = "xaxis_1",
    y_dim: str = "yaxis_1",
    edge: str = "x",
):
    """ Vertically remap a dataset of edge-valued quantities to coarsened pressure levels.

    Args:
        ds (xr.Dataset): input Dataset
        delp (xr.DataArray): pressure thicknesses
        length (xr.DataArray): edge length weights
        coarsening_factor (int): coarsening-factor for pressure levels
        x_dim (str, optional): x-dimension name. Defaults to "xaxis_1"
        y_dim (str, optional): y-dimension name. Defaults to "yaxis_1"
        edge (str, optional): grid cell side to coarse-grain along {"x", "y"}

    Returns:
        xr.Dataset
    """
    grid = create_fv3_grid(
        xr.Dataset(delp),
        x_center="xaxis_1",
        x_outer="xaxis_2",
        y_center="yaxis_2",
        y_outer="yaxis_1",
    )
    delp_edge = grid.interp(delp, edge)
    delp_edge_coarse = edge_weighted_block_average(
        delp_edge, length, coarsening_factor, x_dim=x_dim, y_dim=y_dim, edge=edge
    )
    return _remap_given_delp(
        ds,
        delp_edge,
        delp_edge_coarse,
        length,
        coarsening_factor,
        x_dim=x_dim,
        y_dim=y_dim,
    )


def _remap_given_delp(
    ds,
    delp_fine,
    delp_coarse,
    weights,
    coarsening_factor,
    x_dim: str = "xaxis_1",
    y_dim: str = "yaxis_1",
):
    """Given a fine and coarse delp, do vertical remapping to coarse pressure levels.
    """
    phalf_coarse = thermo.pressure_on_interface(delp_coarse, dim=VERTICAL_DIM)
    phalf_fine = thermo.pressure_on_interface(delp_fine, dim=VERTICAL_DIM)
    phalf_coarse_on_fine = block_upsample(
        phalf_coarse, coarsening_factor, [x_dim, y_dim]
    )

    ds_remap = xr.zeros_like(ds)
    for var in ds:
        ds_remap[var] = remap_levels(phalf_fine, ds[var], phalf_coarse_on_fine)

    masked_weights = weights.where(
        phalf_coarse_on_fine < phalf_fine.max(dim=VERTICAL_DIM), other=0.0,
    )

    return ds_remap, masked_weights


def remap_levels(p_in, f_in, p_out, iv=1, kord=1, dim=VERTICAL_DIM):
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
        dim (str, optional): name of vertical dimension. Defaults to "zaxis_1".

    Returns:
        xr.DataArray: f_in remapped to p_out pressure levels
    """
    if not _mappm_installed:
        raise ImportError("mappm must be installed to use remap_levels")

    dims_except_vertical = list(f_in.dims)
    dims_except_vertical.remove(dim)

    # reshape for mappm
    p_in = p_in.stack(bigdim=dims_except_vertical).transpose()
    f_in = f_in.stack(bigdim=dims_except_vertical).transpose()
    p_out = p_out.stack(bigdim=dims_except_vertical).transpose()

    n_columns = p_in.shape[0]
    assert (
        f_in.shape[0] == n_columns and p_out.shape[0] == n_columns
    ), "All dimensions except vertical must be same size for p_in, f_in and p_out"
    assert (
        f_in.shape[1] == p_in.shape[1] - 1
    ), "f_in must have a vertical dimension one shorter than p_in"

    f_out = xr.zeros_like(f_in)
    # the final argument to mappm is unused by the subroutine
    f_out.values = mappm.mappm(
        p_in.values, f_in.values, p_out.values, 1, n_columns, iv, kord, 0.0
    )

    return f_out.unstack()
