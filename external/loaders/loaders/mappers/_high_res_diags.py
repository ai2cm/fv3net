import numpy as np
import pandas as pd
import vcm
from vcm.cloud import get_fs
import xarray as xr
import zarr.storage as zstore

from .constants import TIME_NAME, TIME_FMT
from ._base import LongRunMapper
from .._utils import standardize_zarr_time_coord


RENAMED_HIGH_RES_DIAG_VARS = {
    "DSWRFtoa_coarse": "DSWRFtoa_prog",
    "DSWRFsfc_coarse": "DSWRFsfc_prog",
    "USWRFtoa_coarse": "USWRFtoa_prog",
    "USWRFsfc_coarse": "USWRFsfc_prog",
    "DLWRFsfc_coarse": "DLWRFsfc_prog",
    "ULWRFtoa_coarse": "ULWRFtoa_prog",
    "ULWRFsfc_coarse": "ULWRFsfc_prog",
    "SHTFLsfc_coarse": "sensible_heat_flux",
    "LHTFLsfc_coarse": "latent_heat_flux",
    "PRATEsfc_coarse": "surface_precipitation_rate","
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
