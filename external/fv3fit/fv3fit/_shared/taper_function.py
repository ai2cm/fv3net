import abc
import numpy as np
import xarray as xr

from fv3fit._shared.novelty_detector import NoveltyDetector


class TaperFunction(abc.ABC):
    @abc.abstractmethod
    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        """
            Given an array of novelty scores (larger scores imply greater novelty),
            determines the multiplicative suppression factor that should be applied
            to the learned tendencies. This factor is a value in [0, 1] for each
            sample, where 0 is total suppression and 1 denotes leaving as is.
        """
        pass


class MaskTaperFunction(TaperFunction):
    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        """
            Completely suppresses the output if the novelty scores is larger that
            some cutoff and does nothing otherwise.
        """
        return xr.where(novelty_score > self.cutoff, 0, 1)

    @classmethod
    def load(
        cls, tapering_config: dict, novelty_detector: NoveltyDetector
    ) -> "MaskTaperFunction":
        if "cutoff" in tapering_config:
            cutoff = tapering_config["cutoff"]
        else:
            cutoff = novelty_detector._get_default_cutoff()
        return cls(cutoff)


class RampTaperFunction(TaperFunction):
    def __init__(self, ramp_min: float, ramp_max: float):
        if ramp_min > ramp_max:
            raise ValueError(f"Ramp max {ramp_max} must be at least min {ramp_min}")
        self.ramp_min = ramp_min
        self.ramp_max = ramp_max

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        """
            Linearly interpolates between complete suppression when the novelty score
            is larger than ramp_max and complete expression when smaller than ramp_min.
        """
        unclipped = (self.ramp_max - novelty_score) / (self.ramp_max - self.ramp_min)
        return np.clip(unclipped, 0, 1)

    @classmethod
    def load(
        cls, tapering_config: dict, novelty_detector: NoveltyDetector
    ) -> "RampTaperFunction":
        if "ramp_min" in tapering_config and "ramp_max" in tapering_config:
            ramp_min, ramp_max = (
                tapering_config["ramp_min"],
                tapering_config["ramp_max"],
            )
        else:
            ramp_min = novelty_detector._get_default_cutoff()
            if ramp_min < 0:
                ramp_max = ramp_min / 2
            elif ramp_min > 0:
                ramp_max = ramp_min * 2
            else:
                ramp_max = 1
        return cls(ramp_min, ramp_max)


class ExponentialDecayTaperFunction(TaperFunction):
    def __init__(self, threshold: float, rate: float):
        self.threshold = threshold
        if rate > 1 or rate < 0:
            raise ValueError(f"Decay rate of {rate} must be in [0, 1].")
        self.rate = rate

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        """
            For novelty scores smaller than threshold, tendencies are completely
            expressed. Otherwise, the fraction of the tendency not suppressed decays
            exponentially with the base rate in [0, 1].
        """
        return np.minimum(self.rate ** (novelty_score - self.threshold), 1)

    @classmethod
    def load(
        cls, tapering_config: dict, novelty_detector: NoveltyDetector
    ) -> "ExponentialDecayTaperFunction":
        if "rate" in tapering_config:
            rate = tapering_config["rate"]
        else:
            rate = 0.5

        if "threshold" in tapering_config:
            threshold = tapering_config["threshold"]
        else:
            threshold = novelty_detector._get_default_cutoff()

        return cls(threshold, rate)
