import os

from .microphysics import Config
from .._utils import wrap_configurable_hook

try:
    config = Config.from_environ(os.environ)
    error = None
except Exception as e:
    config = None
    error = e

microphysics = wrap_configurable_hook(config, error, "microphysics")
