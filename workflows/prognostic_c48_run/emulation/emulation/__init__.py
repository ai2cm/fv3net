import os
from .monitor import Config
from .emulate import microphysics

store = Config.from_environ(os.environ).store
