from typing import Callable
import numpy as np
import xarray as xr
import logging

try:
    import cupy as cp
    import cupy_xarray
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


logger = logging.getLogger(__name__)


def _is_cupy_array(arr):
    return CUPY_AVAILABLE and isinstance(arr, cp.ndarray)

def get_xpy_module(arr):
    return cp if _is_cupy_array(arr) else np


def taper_mask(
    novelty_score: xr.DataArray, cutoff: float = 0, **kwargs
) -> xr.DataArray:
    """
        Completely suppresses the output if the novelty scores is larger that
        some cutoff and does nothing otherwise.
    """
    arr = novelty_score.data
    xp = get_xpy_module(arr)
    logger.info(f"In taper_mask array type and device: {type(arr)} {arr.device} {arr.dtype}")
    return xr.where(novelty_score > cutoff, xp.asarray(0), xp.asarray(1))


def taper_ramp(
    novelty_score: xr.DataArray, ramp_min: float = 0, ramp_max: float = 1, **kwargs
) -> xr.DataArray:
    """
        Linearly interpolates between complete suppression when the novelty score
        is larger than ramp_max and complete expression when smaller than ramp_min.
    """
    xp = get_xpy_module(novelty_score)
    unclipped = (ramp_max - novelty_score) / (ramp_max - ramp_min)
    return xp.clip(unclipped, 0, 1)


def taper_decay(
    novelty_score: xr.DataArray, threshold: float = 0, rate: float = 0.5, **kwargs
) -> xr.DataArray:
    """
        For novelty scores smaller than threshold, tendencies are completely
        expressed. Otherwise, the fraction of the tendency not suppressed decays
        exponentially with the base rate in [0, 1].
    """
    xp = get_xpy_module(novelty_score)
    return xp.minimum(rate ** (novelty_score - threshold), 1)


def get_taper_function(
    name=taper_mask.__name__, config: dict = {}
) -> Callable[[xr.DataArray], xr.DataArray]:
    """
        Returns the proper taper function, based on the configuration. The default is
        the mask tapering function with a cutoff of 0.
    """
    try:
        taper_func = globals()[name]
        return lambda x: taper_func(x, **config)
    except (KeyError):
        raise ValueError("Incorrect tapering name")
