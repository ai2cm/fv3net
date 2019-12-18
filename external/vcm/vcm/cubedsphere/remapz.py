import xarray as xr

from ..calc import thermo
from ..cubedsphere import (
    create_fv3_grid,
    edge_weighted_block_average,
    weighted_block_average,
)
from ..cubedsphere.coarsen import block_upsample
import mappm

# default for restart file
VERTICAL_DIM = "zaxis_1"


def remap_to_area_weighted_pressure(
    ds: xr.Dataset, delp: xr.DataArray, area: xr.DataArray, coarsening_factor: int,
):
    """ docstring """
    delp_coarse = weighted_block_average(
        delp, area, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_2"
    )
    return _remap_given_delp(ds, delp, delp_coarse, area, coarsening_factor)


def remap_to_edge_weighted_pressure(
    ds: xr.Dataset,
    delp: xr.DataArray,
    length: xr.DataArray,
    coarsening_factor: int,
    edge: str = "x",
):
    """ docstring """
    grid = create_fv3_grid(
        xr.Dataset(delp),
        x_center="xaxis_1",
        x_outer="xaxis_2",
        y_center="yaxis_2",
        y_outer="yaxis_1",
    )
    delp_on_edge = grid.interp(delp, edge)
    delp_on_edge_coarse = edge_weighted_block_average(
        delp, length, coarsening_factor, x_dim="xaxis_1", y_dim="yaxis_2", edge=edge
    )
    return _remap_given_delp(
        ds, delp_on_edge, delp_on_edge_coarse, length, coarsening_factor
    )


def _remap_given_delp(ds, delp_fine, delp_coarse, weights, coarsening_factor):
    phalf_coarse = thermo.pressure_on_interface(delp_coarse, dim=VERTICAL_DIM)
    phalf_fine = thermo.pressure_on_interface(delp_fine, dim=VERTICAL_DIM)
    phalf_coarse_on_fine = block_upsample(
        phalf_coarse, coarsening_factor, ["xaxis_1", "yaxis_2"]
    )
    ds_remap = xr.zeros_like(ds)
    for var in ds:
        ds_remap[var] = remap_levels(phalf_fine, ds[var], phalf_coarse_on_fine)
    masked_weights = weights.where(
        phalf_coarse_on_fine < phalf_fine.max(dim=VERTICAL_DIM), other=0.0,
    )
    return ds_remap, masked_weights


def remap_levels(p_in, f_in, p_out, iv=1, kord=1, ptop=0, dim=VERTICAL_DIM):
    dims_except_vertical = list(f_in.dims)
    dims_except_vertical.remove(dim)

    # reshape for mappm
    p_in = p_in.stack(bigdim=dims_except_vertical).transpose()
    f_in = f_in.stack(bigdim=dims_except_vertical).transpose()
    p_out = p_out.stack(bigdim=dims_except_vertical).transpose()

    n_columns = p_in.shape[0]
    npz = f_in.shape[1]
    assert (
        f_in.shape[0] == n_columns and p_out.shape[0] == n_columns
    ), "All dimensions except vertical must be same size for p_in, f_in and p_out"
    assert (
        p_in.shape[1] == npz + 1 and p_out.shape[1] == npz + 1
    ), "f_in must have a vertical dimension one shorter than p_in and p_out"

    f_out = xr.zeros_like(f_in)
    f_out.values = mappm.mappm(
        p_in.values, f_in.values, p_out.values, 1, n_columns, iv, kord, ptop
    )

    return f_out.unstack()
