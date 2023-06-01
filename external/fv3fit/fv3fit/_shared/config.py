import dataclasses
from typing import Any, Hashable, Mapping, Optional, Sequence

# TODO: move all keras configs under fv3fit.keras
import tensorflow as tf
import xarray as xr
import vcm


@dataclasses.dataclass
class TaperConfig:
    cutoff: int
    rate: float
    taper_dim: str = "z"

    def apply(self, data: xr.DataArray) -> xr.DataArray:
        n_levels = len(data[self.taper_dim])
        scaling = xr.DataArray(
            vcm.vertical_tapering_scale_factors(
                n_levels=n_levels, cutoff=self.cutoff, rate=self.rate
            ),
            dims=[self.taper_dim],
        )
        return scaling * data


@dataclasses.dataclass
class CacheConfig:
    """
    Attributes:
        local_download_path: location to save data locally
        in_memory: if True, keep data in memory once loaded
    """

    local_download_path: Optional[str] = None
    in_memory: bool = False


@dataclasses.dataclass
class LearningRateScheduleConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        cls = getattr(tf.keras.optimizers.schedules, self.name)
        return cls(**self.kwargs)


@dataclasses.dataclass
class OptimizerConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    learning_rate_schedule: Optional[LearningRateScheduleConfig] = None

    @property
    def instance(self) -> tf.keras.optimizers.Optimizer:
        cls = getattr(tf.keras.optimizers, self.name)
        kwargs = dict(**self.kwargs)

        if self.learning_rate_schedule:
            kwargs["learning_rate"] = self.learning_rate_schedule.instance

        return cls(**kwargs)

    def __post_init__(self):
        if "learning_rate" in self.kwargs and self.learning_rate_schedule is not None:
            raise ValueError(
                "Learning rate ambiguity from kwargs and learning rate schedule set"
                " in OptimizerConfig."
            )


@dataclasses.dataclass
class RegularizerConfig:
    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self) -> Optional[tf.keras.regularizers.Regularizer]:
        if self.name.lower() != "none":
            cls = getattr(tf.keras.regularizers, self.name)
            instance = cls(**self.kwargs)
        else:
            instance = None
        return instance


@dataclasses.dataclass
class SliceConfig:
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    @property
    def slice(self):
        return slice(self.start, self.stop, self.step)


@dataclasses.dataclass(frozen=True)
class PackerConfig:
    """
    Configuration for packing.

    Attributes:
        clip: a mapping from variable name to configuration for the slice of
            the feature (last) dimension of that variable we want to retain.
            Used to exclude data (e.g. at start or end of dimension). User
            must ensure the last dimension is the dimension they want to clip.
    """

    clip: Mapping[Hashable, SliceConfig] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class SquashedOutputConfig:
    """"
    Configuration of output squashing

    Attributes:
        squash_by_name: name of the variable that will determine whether outputs
            are squashed
        additional_squash_target_names: name of the variables to be squashed in
            addition to `squash_by_name`
        squash_threshold: threshold value in squash_by_name below which squashing will
            occur for this sample and feature position for all target variables
        squash_to: value to which squashed values will be set
    """

    squash_by_name: Hashable
    squash_threshold: float
    squash_to: float = 0.0
    additional_squash_target_names: Sequence[Hashable] = ()

    def squash(self, predictions: xr.Dataset) -> xr.Dataset:
        squashed_predictions = predictions.copy()
        for name in [self.squash_by_name] + list(self.additional_squash_target_names):
            squashed_predictions[name] = predictions[name].where(
                predictions[self.squash_by_name] > self.squash_threshold,
                self.squash_to,
            )
        return squashed_predictions
