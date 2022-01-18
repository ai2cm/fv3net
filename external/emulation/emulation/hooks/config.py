import dataclasses
import logging
from typing import Optional

import dacite
import yaml
from emulation.hooks import microphysics, monitor
from emulation.hooks.microphysics import _load_tf_model

logger = logging.getLogger("emulation")


def error_on_call(e):
    def func(state):
        raise e

    return func


def do_nothing(state):
    pass


@dataclasses.dataclass
class ModelConfig:
    path: str


@dataclasses.dataclass
class EmulationConfig:
    model: Optional[ModelConfig] = None
    storage: Optional[monitor.StorageConfig] = None

    def build_model_hook(self):
        if self.model is None:
            return error_on_call(
                ValueError("model not defined. Check the configuration.")
            )
        else:
            return microphysics.MicrophysicsHook(_load_tf_model(self.model.path))

    def build_storage_hook(self):
        if self.storage is None:
            logger.info("No storage configured.")
            return do_nothing
        else:
            return monitor.StorageHook(**dataclasses.asdict(self.storage))


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
