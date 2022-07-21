from typing import Callable
import numpy as np
import xarray as xr

_MASK_NAME = "mask"
_RAMP_NAME = "ramp"
_DECAY_NAME = "decay"


def get_taper_function(
    name=_MASK_NAME, config: dict = {}
) -> Callable[[xr.DataArray], xr.DataArray]:
    if name == _MASK_NAME:
        return lambda x: taper_mask(x, **config)
    elif name == _RAMP_NAME:
        return lambda x: taper_ramp(x, **config)
    elif name == _DECAY_NAME:
        return lambda x: taper_decay(x, **config)
    else:
        raise ValueError("Incorrect tapering name")


def taper_mask(
    novelty_score: xr.DataArray, cutoff: float = 0, **kwargs
) -> xr.DataArray:
    """
        Completely suppresses the output if the novelty scores is larger that
        some cutoff and does nothing otherwise.
    """
    return xr.where(novelty_score > cutoff, 0, 1)


def taper_ramp(
    novelty_score: xr.DataArray, ramp_min: float = 0, ramp_max: float = 1, **kwargs
) -> xr.DataArray:
    """
        Linearly interpolates between complete suppression when the novelty score
        is larger than ramp_max and complete expression when smaller than ramp_min.
    """
    unclipped = (ramp_max - novelty_score) / (ramp_max - ramp_min)
    return np.clip(unclipped, 0, 1)


def taper_decay(
    novelty_score: xr.DataArray, threshold: float = 0, rate: float = 0.5, **kwargs
) -> xr.DataArray:
    """
        For novelty scores smaller than threshold, tendencies are completely
        expressed. Otherwise, the fraction of the tendency not suppressed decays
        exponentially with the base rate in [0, 1].
    """
    return np.minimum(rate ** (novelty_score - threshold), 1)
