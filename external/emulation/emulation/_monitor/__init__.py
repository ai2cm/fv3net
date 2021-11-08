import os

from .monitor import StorageHook


try:
    _config = StorageHook.from_environ(os.environ)
    store = _config.store
except (KeyError, FileNotFoundError) as e:
    _config = None

    error = f"The StorageHook config could not be initialized" f" due to error: {e}"

    def store(state):
        raise ImportError(error)
