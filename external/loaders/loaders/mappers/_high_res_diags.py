from vcm.cloud import get_fs
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


def open_high_res_diags(url):
    fs = get_fs(url)
    mapper = fs.get_mapper(url)
    consolidated = True if ".zmetadata" in mapper else False
    ds = xr.open_zarr(zstore.LRUStoreCache(mapper, 1024), consolidated=consolidated)
    return ds


class HighResDiags(LongRunMapper):
    def __init__(
        self,
        ds,
        renamed_vars=RENAMED_HIGH_RES_DIAG_VARS,
    ):
        self.ds = standardize_zarr_time_coord(ds).rename(RENAMED_HIGH_RES_DIAG_VARS)
