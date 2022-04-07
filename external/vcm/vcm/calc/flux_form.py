from typing import Tuple
import dask
import xarray as xr
import vcm


def _tendency_to_flux(
    tendency: xr.DataArray,
    toa_net_flux: xr.DataArray,
    surface_upward_flux: xr.DataArray,
    delp: xr.DataArray,
    dim: str = "z",
    rectify: bool = True,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Compute cell-interface fluxes from cell-center tendencies and boundary fluxes.

    Args:
        tendency: Vertical column of tendencies.
        toa_net_flux: Net flux at TOA.
        surface_upward_flux: Upward flux at surface.
        delp: pressure thickness in Pa.
        dim: (optional) name of vertical dimension.
        rectify: (optional) whether to force downward surface flux to be positive.

    Returns:
        tuple of array of column of net fluxes and downward surface flux. The column
        of net fluxes represents flux at cell interfaces above each cell center.

    See also:
        https://github.com/ai2cm/explore/blob/master/oliwm/2021-12-13-fine-res-in-flux-
        form/2021-12-13-fine-res-in-flux-form-proposal-v2.ipynb
    """
    flux = -vcm.mass_cumsum(tendency, delp, dim=dim)
    flux = flux.pad({dim: (1, 0)}, constant_values=0.0)
    flux += toa_net_flux
    downward_sfc_flux = flux.isel({dim: -1}) + surface_upward_flux
    if rectify:
        downward_sfc_flux = downward_sfc_flux.where(downward_sfc_flux >= 0, 0)
    flux = flux.isel({dim: slice(None, -1)})
    if isinstance(flux.data, dask.array.Array):
        flux = flux.chunk({dim: flux.sizes[dim]})
    return flux, downward_sfc_flux


def _tendency_to_implied_surface_downward_flux(
    tendency: xr.DataArray,
    toa_net_flux: xr.DataArray,
    surface_upward_flux: xr.DataArray,
    delp: xr.DataArray,
    dim: str = "z",
    rectify: bool = True,
) -> xr.DataArray:
    """Compute implied downward surface flux assuming budget closure.

    Args:
        tendency: Vertical column of tendencies.
        toa_net_flux: Net flux at TOA.
        surface_upward_flux: Upward flux at surface.
        delp: pressure thickness in Pa.
        dim: (optional) name of vertical dimension.
        rectify: (optional) whether to force downward surface flux to be positive.

    Returns:
        downward surface flux assuming column-integral of tendencies equals sum of
        TOA net flux and surface net flux.
    """
    column_integrated_tendency = vcm.mass_integrate(tendency, delp, dim=dim)
    # <tendency> = TOA_net_flux - surface_down_flux + surface_upward_flux, or
    # surface_down_flux = TOA_net_flux + surface_upward_flux - <tendency>
    downward_sfc_flux = toa_net_flux + surface_upward_flux - column_integrated_tendency
    if rectify:
        downward_sfc_flux = downward_sfc_flux.where(downward_sfc_flux >= 0, 0)
    return downward_sfc_flux


def _flux_to_tendency(
    net_flux: xr.DataArray,
    surface_downward_flux: xr.DataArray,
    surface_upward_flux: xr.DataArray,
    delp: xr.DataArray,
    dim: str = "z",
) -> xr.DataArray:
    """Compute tendencies given flux between model levels and surface fluxes.

    Args:
        net_flux: The net flux between model layers, not including surface interface.
        surface_downward_flux: Downward flux at surface.
        surface_upward_flux: Upward flux at surface.
        delp: pressure thickness in Pa.
        dim: (optional) name of vertical dimension.

    Returns:
        tendency at cell centers.
    """
    surface_net_flux = surface_downward_flux - surface_upward_flux
    net_flux_full = xr.concat([net_flux, surface_net_flux], dim="z")
    tendency = -vcm.mass_divergence(
        net_flux_full, delp, dim_center=dim, dim_interface=dim
    )
    return tendency
