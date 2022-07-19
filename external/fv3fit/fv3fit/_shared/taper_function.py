import abc
import xarray as xr

from fv3fit._shared.novelty_detector import NoveltyDetector


class TaperFunction(abc.ABC):
    @abc.abstractmethod
    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        pass

    @classmethod
    def load(cls, tapering_config: dict, novelty_detector: NoveltyDetector):
        name = tapering_config["name"]
        if name == MaskTaperFunction._NAME:
            return MaskTaperFunction.load(tapering_config, novelty_detector)
        elif name == RampTaperFunction._NAME:
            return RampTaperFunction.load(tapering_config, novelty_detector)
        elif name == ExponentialDecayTaperFunction._NAME:
            return ExponentialDecayTaperFunction.load(tapering_config, novelty_detector)
        else:
            return None


class MaskTaperFunction(TaperFunction):
    _NAME = "mask"

    def __init__(self, cutoff: float):
        self.cutoff = cutoff

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        return xr.where(novelty_score > self.cutoff, 1, 0)

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
    _NAME = "ramp"

    def __init__(self, ramp_min: float, ramp_max: float):
        assert ramp_min <= ramp_max
        self.ramp_min = ramp_min
        self.ramp_max = ramp_max

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        unclipped = (self.ramp_max - novelty_score) / (self.ramp_max - self.ramp_min)
        return xr.ufuncs.minimum(xr.ufuncs.maximum(unclipped, 1), 0)

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
    _NAME = "decay"

    def __init__(self, threshold: float, rate: float):
        self.threshold = threshold
        assert rate >= 0 and rate <= 1
        self.rate = rate

    def get_taper_value(self, novelty_score: xr.DataArray) -> xr.DataArray:
        return xr.ufuncs.maximum(self.rate ** (novelty_score - self.threshold), 1)

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
