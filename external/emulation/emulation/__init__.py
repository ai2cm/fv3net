import os

from emulation.hooks import MicrophysicsHook, StorageHook

try:
    _config = MicrophysicsHook.from_environ(os.environ)
    microphysics = _config.microphysics
except (KeyError, FileNotFoundError) as e:
    _config = None

    error = (
        f"The MicrophysicsHook config could not be initialized" f" due to error: {e}"
    )

    def microphysics(state):
        raise ImportError(error)


try:
    _config = StorageHook.from_environ(os.environ)
    store = _config.store
except (KeyError, FileNotFoundError) as e:
    _config = None

    error = f"The StorageHook config could not be initialized" f" due to error: {e}"

    def store(state):
        raise ImportError(error)
