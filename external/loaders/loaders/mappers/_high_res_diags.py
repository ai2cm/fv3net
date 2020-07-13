from typing import Mapping
from vcm.cloud import get_fs
from vcm import safe
import xarray as xr
import zarr.storage as zstore
from .._utils import net_precipitation_from_physics, net_heating_from_physics
from ._base import LongRunMapper
from ..constants import RENAMED_SHIELD_DIAG_VARS

RENAMED_SHIELD_DIMS = {
    "grid_xt": "x",
    "grid_yt": "y",
}


def open_high_res_diags(
    url: str,
    renamed_vars: Mapping = RENAMED_SHIELD_DIAG_VARS,
    renamed_dims: Mapping = RENAMED_SHIELD_DIMS,
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
    ds = (
        xr.open_zarr(zstore.LRUStoreCache(mapper, 1024), consolidated=consolidated)
        .rename({**renamed_vars, **renamed_dims})
        .pipe(safe.get_variables, renamed_vars.values())
        .assign_coords({"tile": range(6)})
        .pipe(_insert_net_terms)
    )

    return LongRunMapper(ds)


def _insert_net_terms(ds: xr.Dataset) -> xr.Dataset:

    return ds.assign(
        {
            "net_heating": net_heating_from_physics(ds),
            "net_precipitation": net_precipitation_from_physics(ds),
        }
    )
