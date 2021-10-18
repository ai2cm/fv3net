import os

from .monitor import Config
from .._utils import wrap_configurable_hook

try:
    config = Config.from_environ(os.environ)
    error = None
except Exception as e:
    config = None
    error = e

store = wrap_configurable_hook(config, error, "store")
