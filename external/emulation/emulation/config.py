import dataclasses
import datetime
import logging
from typing import Iterable, Mapping, Optional, Tuple
import os

import cftime
import dacite
import f90nml
import yaml
from emulation._emulate.microphysics import (
    MicrophysicsHook,
    IntervalSchedule,
    Mask,
    TimeMask,
)
from emulation._monitor.monitor import StorageConfig, StorageHook
from emulation._time import from_datetime, to_datetime
from emulation.masks import RangeMask, LevelMask, compose_masks
import emulation.zhao_carr

logger = logging.getLogger("emulation")


def do_nothing(state):
    pass


def _get_timestep(namelist):
    return int(namelist["coupler_nml"]["dt_atmos"])


def _load_nml():
    path = os.path.join(os.getcwd(), "input.nml")
    namelist = f90nml.read(path)
    logger.info(f"Loaded namelist for ZarrMonitor from {path}")

    return namelist


@dataclasses.dataclass
class Range:
    min: Optional[float] = None
    max: Optional[float] = None


@dataclasses.dataclass
class LevelSlice:
    start: Optional[int] = None
    stop: Optional[int] = None


@dataclasses.dataclass
class ModelConfig:
    """

    Attributes:
        path: path to a saved tensorflow model
        online_schedule: an object controlling when to use the emulator instead
            of the fortran model. Only supports scheduling by an interval.
            The physics is used for the first half of the interval, and the ML
            for the second half.
        ranges: post-hoc limits to apply to the predicted values
        mask_emulator_levels:  override the emulator tendencies with the fortran
            physics tendencies for a specified level range.
        cloud_squash: all cloud values less than this amount (including
            negative values) will be squashed to zero.
        gscond_cloud_conservative: infer the gscond cloud from conservation via
            gscond humidity tendency
        enforce_conservative: if True, temperature and humidity change will be
            inferred from the cloud change after all masks have been applied. The
            latent heat is inferred assuming liquid condensate. Differs from,
            but typically used in concert with ``gscond_cloud_conservative``.

    """

    path: str
    online_schedule: Optional[IntervalSchedule] = None
    ranges: Mapping[str, Range] = dataclasses.field(default_factory=dict)
    mask_emulator_levels: Mapping[str, LevelSlice] = dataclasses.field(
        default_factory=dict
    )
    cloud_squash: Optional[float] = None
    gscond_cloud_conservative: bool = False
    mask_gscond_identical_cloud: bool = False
    mask_gscond_zero_cloud: bool = False
    enforce_conservative: bool = False

    def build(self) -> MicrophysicsHook:
        return MicrophysicsHook(self.path, mask=self._build_mask())

    def _build_mask(self) -> Mask:
        return compose_masks(self._build_masks())

    def _build_masks(self) -> Iterable[Mask]:
        if self.online_schedule:
            yield TimeMask(self.online_schedule)

        for key, range in self.ranges.items():
            yield RangeMask(key, min=range.min, max=range.max)

        if self.gscond_cloud_conservative:
            yield emulation.zhao_carr.infer_gscond_cloud_from_conservation

        if self.cloud_squash is not None:
            yield lambda x, y: emulation.zhao_carr.squash_gscond(
                x, y, self.cloud_squash
            )
            yield lambda x, y: emulation.zhao_carr.squash_precpd(
                x, y, self.cloud_squash
            )

        if self.mask_gscond_identical_cloud:
            yield emulation.zhao_carr.mask_where_fortran_cloud_identical

        if self.mask_gscond_zero_cloud:
            yield emulation.zhao_carr.mask_where_fortran_cloud_vanishes_gscond

        if self.enforce_conservative:
            yield emulation.zhao_carr.enforce_conservative_gscond

        for key, _slice in self.mask_emulator_levels.items():
            yield LevelMask(key, start=_slice.start, stop=_slice.stop)

    @staticmethod
    def from_dict(dict_: dict) -> "ModelConfig":
        return dacite.from_dict(ModelConfig, dict_, config=dacite.Config(strict=True))


@dataclasses.dataclass
class EmulationConfig:
    model: Optional[ModelConfig] = None
    gscond: Optional[ModelConfig] = None
    storage: Optional[StorageConfig] = None

    @staticmethod
    def _build_model(model: ModelConfig):
        if model is None:
            logger.info("No model configured.")
            return do_nothing
        else:
            return model.build().microphysics

    def build_model_hook(self):
        return self._build_model(self.model)

    def build_gscond_hook(self):
        return self._build_model(self.gscond)

    def build_storage_hook(self):
        hook = _get_storage_hook(self.storage)
        return hook.store if hook else do_nothing

    @staticmethod
    def from_dict(dict_: dict) -> "EmulationConfig":
        return dacite.from_dict(
            EmulationConfig,
            dict_,
            config=dacite.Config(
                type_hooks={
                    cftime.DatetimeJulian: from_datetime,
                    datetime.timedelta: lambda x: datetime.timedelta(seconds=x),
                }
            ),
        )

    def to_dict(self) -> dict:
        def factory(keyvals):
            out = {}
            for key, val in keyvals:
                if isinstance(val, cftime.DatetimeJulian):
                    out[key] = to_datetime(val)
                elif isinstance(val, datetime.timedelta):
                    out[key] = int(val.total_seconds())
                else:
                    out[key] = val
            return out

        return dataclasses.asdict(self, dict_factory=factory)


def _get_storage_hook(storage_config: Optional[StorageConfig]) -> Optional[StorageHook]:

    if storage_config is None:
        logger.info("No storage configured.")
        return None

    try:
        namelist = _load_nml()
    except FileNotFoundError:
        logger.warn("Namelist could not be loaded. Storage disabled.")
        return None

    # get metadata
    path = os.getenv("VAR_META_PATH", storage_config.var_meta_path)
    try:
        with open(str(path), "r") as f:
            variable_metadata = yaml.safe_load(f)
            logger.info(f"Loaded variable metadata from: {path}")
    except FileNotFoundError:
        variable_metadata = {}
        logger.info(f"No metadata found at: {path}")

    timestep = _get_timestep(namelist)
    layout: Tuple[int, int] = namelist["fv_core_nml"]["layout"]

    kwargs = dataclasses.asdict(storage_config)
    kwargs.pop("var_meta_path", None)
    return StorageHook(
        metadata=variable_metadata, layout=layout, dt_sec=timestep, **kwargs
    )


def get_hooks():
    path = "fv3config.yml"
    config_key = "zhao_carr_emulation"
    try:
        with open(path) as f:
            dict_ = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warn("Config not found...using defaults.")
        dict_ = {}
    config = EmulationConfig.from_dict(dict_.get(config_key, {}))

    return (
        config.build_gscond_hook(),
        config.build_model_hook(),
        config.build_storage_hook(),
    )
