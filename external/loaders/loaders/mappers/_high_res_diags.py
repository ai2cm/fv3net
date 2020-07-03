from typing import Mapping
from vcm.cloud import get_fs
from vcm import safe
import xarray as xr
import zarr.storage as zstore

from ._base import LongRunMapper


RENAMED_HIGH_RES_DIAG_VARS = {
    "DSWRFtoa_coarse": "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "DSWRFsfc_coarse": "total_sky_downward_shortwave_flux_at_surface",
    "USWRFtoa_coarse": "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "USWRFsfc_coarse": "total_sky_upward_shortwave_flux_at_surface",
    "DLWRFsfc_coarse": "total_sky_downward_longwave_flux_at_surface",
    "ULWRFtoa_coarse": "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "ULWRFsfc_coarse": "total_sky_upward_longwave_flux_at_surface",
    "SHTFLsfc_coarse": "sensible_heat_flux",
    "LHTFLsfc_coarse": "latent_heat_flux",
    "PRATEsfc_coarse": "surface_precipitation_rate",
}

RENAMED_HIGH_RES_DIMS = {
    "grid_xt": "x",
    "grid_yt": "y",
}


def open_high_res_diags(
    url: str,
    renamed_vars: Mapping = RENAMED_HIGH_RES_DIAG_VARS,
    renamed_dims: Mapping = RENAMED_HIGH_RES_DIMS,
) -> LongRunMapper:
    """Create a mapper for SHiELD 2D diagnostics data.
    Handles renaming to state variable names.

    Args:
        url (str): path to diagnostics zarr
        renamed_vars (Mapping, optional): Defaults to RENAMED_HIGH_RES_DIAG_VARS.
        renamed_dims (Mapping, optional): Defaults to RENAMED_HIGH_RES_DIMS.

    Returns:
        LongRunMapper
    """

    fs = get_fs(url)
    mapper = fs.get_mapper(url)
    consolidated = True if ".zmetadata" in mapper else False
    ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024), consolidated=consolidated)
    ds = safe.get_variables(
        ds.rename({**renamed_vars, **renamed_dims}), RENAMED_HIGH_RES_DIAG_VARS.values()
    )
    ds = ds.assign_coords({"tile": range(6)})
    return LongRunMapper(ds)
