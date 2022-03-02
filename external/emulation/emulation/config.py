import dataclasses
import logging
from typing import Callable, Optional

import dacite
import yaml
from emulation._emulate.microphysics import MicrophysicsHook
from emulation._emulate.mask import MaskConfig
from emulation._monitor.monitor import StorageConfig, StorageHook
from emulation._typing import FortranState


logger = logging.getLogger("emulation")

StateFunc = Callable[[FortranState], None]


def do_nothing(state: FortranState) -> None:
    pass


@dataclasses.dataclass
class ModelConfig:
    path: str
    mask: MaskConfig = MaskConfig()


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
            hook = MicrophysicsHook(model.path, model.mask)
            return hook.microphysics

    def build_model_hook(self):
        return self._build_model(self.model)

    def build_gscond_hook(self):
        return self._build_model(self.gscond)

    def build_storage_hook(self) -> StateFunc:
        if self.storage is None:
            logger.info("No storage configured.")
            return do_nothing
        else:
            return StorageHook(**dataclasses.asdict(self.storage)).store


def get_hooks():
    path = "fv3config.yml"
    config_key = "zhao_carr_emulation"
    try:
        with open(path) as f:
            dict_ = yaml.safe_load(f)
    except FileNotFoundError:
        logging.warn("Config not found...using defaults.")
        dict_ = {}
    config = dacite.from_dict(EmulationConfig, dict_.get(config_key, {}))

    return (
        config.build_gscond_hook(),
        config.build_model_hook(),
        config.build_storage_hook(),
    )
