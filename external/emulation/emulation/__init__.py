# flake8: noqa
# Tensorflow looks at sys args which are not initialized
# when this module is loaded under callpyfort, so ensure
# it's available here
import sys

if not hasattr(sys, "argv"):
    sys.argv = [""]

from emulation.config import (
    get_hooks,
    EmulationConfig,
    ModelConfig,
    StorageConfig,
)


gscond, microphysics, store = get_hooks()
