from typing import Mapping
from vcm.cloud import get_fs
from vcm import safe
import xarray as xr
import zarr.storage as zstore

from ._base import LongRunMapper
from .._utils import standardize_zarr_time_coord


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
        url,
        renamed_vars: Mapping = RENAMED_HIGH_RES_DIAG_VARS,
        renamed_dims: Mapping = RENAMED_HIGH_RES_DIMS
):
    fs = get_fs(url)
    mapper = fs.get_mapper(url)
    consolidated = True if ".zmetadata" in mapper else False
    ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024), consolidated=consolidated)
    ds = standardize_zarr_time_coord(ds)
    ds = safe.get_variables(
        ds.rename({**renamed_vars, **renamed_dims}),
        RENAMED_HIGH_RES_DIAG_VARS.values())
    ds = ds.assign_coords({"tile": range(6)})
    return LongRunMapper(ds)
