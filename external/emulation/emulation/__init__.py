from emulation.hooks.config import (
    get_hooks,
    EmulationConfig,
    ModelConfig,
)
from emulation.hooks.monitor import StorageConfig
from emulation.logging import setup_logging


microphysics, store = get_hooks()
setup_logging()
