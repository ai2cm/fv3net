import os

from .monitor import Config

store = Config.from_environ(os.environ).store
