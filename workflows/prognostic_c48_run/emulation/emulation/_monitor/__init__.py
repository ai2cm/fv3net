import os

from .monitor import Config

try:
    config = Config.from_environ(os.environ)
except KeyError as e:
    config = None
    error = e


# The configuration during import creates a chicken-egg situation
# for testing, this allows us to test and inject a config
def store(state):
    if config is None:
        raise ImportError(
            f"Monitor store could not be initialized due to error: {error}"
        )

    return config.store(state)
