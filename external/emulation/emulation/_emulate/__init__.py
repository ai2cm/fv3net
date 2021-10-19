import os

from .microphysics import MicrophysicsHook

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
