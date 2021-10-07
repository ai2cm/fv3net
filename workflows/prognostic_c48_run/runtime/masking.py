import xarray as xr
from functools import partial
from typing import (
    Callable,
    Hashable,
    Mapping,
    Optional,
)

from runtime.names import EAST_WIND, SPHUM, CLOUD


__all__ = ["get_mask", "where_masked"]


def where_masked(
    left: Mapping[Hashable, xr.DataArray],
    right: Mapping[Hashable, xr.DataArray],
    compute_mask: Callable[[Hashable, xr.DataArray], xr.DataArray],
) -> Mapping[Hashable, xr.DataArray]:
    """Blend two states based on a mask

    Where ``compute_mask(left[name], name)`` is True return ``left``, otherwise
    return ``right``.
    """
    updated_state = dict(left)
    for key in right:
        arr = left[key]
        mask = compute_mask(key, arr)
        updated_state[key] = arr.where(mask, right[key].variable)
    return updated_state


def get_mask(kind: str, ignore_humidity_below: Optional[int] = None):
    if kind == "default":
        return partial(
            compute_mask_default, ignore_humidity_below=ignore_humidity_below
        )
    else:
        return eval(f"compute_mask_{kind}")


def compute_mask_default(
    name: Hashable, arr: xr.DataArray, ignore_humidity_below: Optional[int] = None
) -> xr.DataArray:
    if name == SPHUM:
        if ignore_humidity_below is not None:
            return arr.z < ignore_humidity_below
        else:
            return xr.DataArray(False)
    else:
        return xr.DataArray(False)


def compute_mask_2021_09_16(name: Hashable, arr: xr.DataArray) -> xr.DataArray:
    """The mask proposed in the emulation track log on Sept 16.
    """
    if name == SPHUM:
        return arr.z < 20
    elif name == EAST_WIND:
        return arr.z < 6
    else:
        return xr.DataArray(False)


def compute_mask_no_cloud(name: Hashable, arr: xr.DataArray) -> xr.DataArray:
    """Ignore cloud water outputs from emulator
    """
    if name == CLOUD:
        return xr.DataArray(True)
    else:
        return xr.DataArray(False)


def compute_mask_no_cloud_no_qv(name: Hashable, arr: xr.DataArray) -> xr.DataArray:
    """Ignore cloud water outputs from emulator
    """
    if name in [CLOUD, SPHUM]:
        return xr.DataArray(True)
    else:
        return xr.DataArray(False)


def compute_mask_no_sphum_bl(name: Hashable, arr: xr.DataArray) -> xr.DataArray:
    if name == SPHUM:
        return (arr.z > 68) & (arr.z < 20)
    else:
        return xr.DataArray(False)


def compute_mask_no_sphum_bl_no_cloud(
    name: Hashable, arr: xr.DataArray
) -> xr.DataArray:
    if name == SPHUM:
        return (arr.z > 68) & (arr.z < 20)
    elif name == CLOUD:
        return xr.DataArray(True)
    else:
        return xr.DataArray(False)
