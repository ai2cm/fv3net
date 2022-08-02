import logging
import xarray as xr
from typing import Mapping, Any

logger = logging.getLogger(__name__)


def open_zarr(url: str, **kwargs: Mapping[Any, Any]) -> xr.Dataset:
    cachedir = "/tmp/files"
    logger.info(f"Opening {url} with caching at {cachedir}.")
    return xr.open_zarr(
        "filecache::" + url,
        storage_options={"filecache": {"cache_storage": cachedir}},
        **kwargs,
    )
