from emulation.hooks.config import (
    get_hooks,
    EmulationConfig,
    ModelConfig,
)
from emulation.hooks.monitor import StorageConfig


microphysics, store = get_hooks()
