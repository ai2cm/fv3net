import dataclasses
import logging
from typing import Optional

import dacite
import yaml
from emulation._emulate.microphysics import MicrophysicsHook
from emulation._monitor.monitor import StorageConfig, StorageHook

logger = logging.getLogger("emulation")


def do_nothing(state):
    pass


@dataclasses.dataclass
class ModelConfig:
    path: str


@dataclasses.dataclass
class EmulationConfig:
    model: Optional[ModelConfig] = None
    storage: Optional[StorageConfig] = None

    def build_model_hook(self):
        if self.model is None:
            logger.info("No model configured.")
            return do_nothing
        else:
            return MicrophysicsHook(self.model.path).microphysics

    def build_storage_hook(self):
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

    return config.build_model_hook(), config.build_storage_hook()
