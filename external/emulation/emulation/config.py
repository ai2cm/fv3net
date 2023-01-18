import dataclasses
import datetime
import logging
from typing import Iterable, Mapping, Optional, Tuple, Union, List
import os

import cftime
import dacite
import f90nml
import tensorflow as tf
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
import emulation._filesystem
import emulation.models
import emulation.zhao_carr

from fv3fit.train_microphysics import (
    TransformT,
    ComposedTransformFactory,
    TensorTransform,
)

logger = logging.getLogger("emulation")


def _load_tf_model(model_path: str) -> tf.keras.Model:
    logger.info(f"Loading keras model: {model_path}")
    with emulation._filesystem.get_dir(model_path) as local_model_path:
        return tf.keras.models.load_model(local_model_path)


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
    """

    Attributes:
        fill_value: how to fill the values between start and stop. If a float,
            then fill with fill_value. If a string, then fill with the values from
            ``truth[fill_value]``.
    """

    start: Optional[int] = None
    stop: Optional[int] = None
    fill_value: Union[float, str, None] = None


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
        mask_emulator_levels:  levels to mask the emulator tendencies at.
        cloud_squash: all cloud values less than this amount (including
            negative values) will be squashed to zero.
        gscond_cloud_conservative: infer the gscond cloud from conservation via
            gscond humidity tendency
        enforce_conservative: if True, temperature and humidity change will be
            inferred from the cloud change after all masks have been applied. The
            latent heat is inferred assuming liquid condensate. Differs from,
            but typically used in concert with ``gscond_cloud_conservative``.
        enforce_conservative_phase_dependent: Same as ``enforce_conservative``
            but uses temperature and cloud dependent latent heat value similar
            to the gscond.f.
        tensor_transform: differerentiable tensorflow
            transformations to apply before and after data is passed to models.
            Currently only works with transforms that do not require to be
            built.
        batch_size: number of columns to batch the ml prediction by. May reduce
            memory use.
        enforce_strict_precpd_conservative: Use a conservation to determine total precip
            from clouds, and then update evaporation and temperature based on
            available precip at each vertical level (considering fluxes from above).
            Ensures precipitation values are never negative and that
            evaporation is limited to the amount of available precip in each cell.
            Final temperature values are inferred from the adjusted evaporation
            in this case.
        simple_precip_conservative: A precip conservation method which sums total water
            change in cloud and vapor without consideration for limitations
    """

    path: Optional[str] = None
    classifier_path: Optional[str] = None
    online_schedule: Optional[IntervalSchedule] = None
    tensor_transform: List[TransformT] = dataclasses.field(default_factory=list)
    ranges: Mapping[str, Range] = dataclasses.field(default_factory=dict)
    mask_emulator_levels: Mapping[str, LevelSlice] = dataclasses.field(
        default_factory=dict
    )
    cloud_squash: Optional[float] = None
    gscond_cloud_conservative: bool = False
    mask_gscond_identical_cloud: bool = False
    mask_gscond_zero_cloud: bool = False
    enforce_conservative: bool = False
    enforce_conservative_phase_dependent: bool = False
    mask_gscond_zero_cloud_classifier: bool = False
    mask_gscond_no_tend_classifier: bool = False
    mask_precpd_zero_cloud_classifier: bool = False
    enforce_strict_precpd_conservative: bool = False
    simple_precip_conservative: bool = False
    batch_size: int = 512
    input_passthru: bool = False

    def __post_init__(self):
        if self.enforce_conservative and self.enforce_conservative_phase_dependent:
            raise ValueError("These options are mutually exclusive.")

        if self.enforce_strict_precpd_conservative and self.simple_precip_conservative:
            raise ValueError("Conservative precip flags should not both be true.")

    @property
    def _transform_factory(self) -> ComposedTransformFactory:
        return ComposedTransformFactory(self.tensor_transform)

    def _build_transform(self) -> TensorTransform:
        return self._transform_factory.build({})

    def build(self) -> MicrophysicsHook:
        if self.path:
            regressor = _load_tf_model(self.path)
            classifier = (
                _load_tf_model(self.classifier_path)
                if self.classifier_path is not None
                else None
            )
            adapter = emulation.models.combine_classifier_and_regressor(
                classifier, regressor, self.batch_size
            )
        else:

            def adapter(x):
                return x

        transform = self._build_transform()
        transformed_model = emulation.models.transform_model(adapter, transform)
        return MicrophysicsHook(model=transformed_model, mask=self._build_mask())

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

        if self.mask_gscond_no_tend_classifier:
            yield emulation.zhao_carr.mask_zero_tend_classifier

        if self.mask_gscond_zero_cloud_classifier:
            yield emulation.zhao_carr.mask_zero_cloud_classifier

        if self.mask_precpd_zero_cloud_classifier:
            yield emulation.zhao_carr.mask_zero_cloud_classifier_precpd

        if self.enforce_conservative:
            yield emulation.zhao_carr.enforce_conservative_gscond
        elif self.enforce_conservative_phase_dependent:
            yield emulation.zhao_carr.enforce_conservative_phase_dependent

        if self.simple_precip_conservative:
            yield emulation.zhao_carr.conservative_precip_simple
        elif self.enforce_strict_precpd_conservative:
            yield emulation.zhao_carr.enforce_conservative_precpd

        for key, _slice in self.mask_emulator_levels.items():
            yield LevelMask(
                key, start=_slice.start, stop=_slice.stop, fill_value=_slice.fill_value
            )

        if self.input_passthru:
            yield emulation.zhao_carr.preserve_state_inputs


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
                },
                strict=True,
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
