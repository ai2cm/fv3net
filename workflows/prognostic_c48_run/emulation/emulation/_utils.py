from typing import Callable


def wrap_configurable_hook(config, error, func_name) -> Callable:
    """
    This function wraps the call-py-fort hooks to fail when the
    functions are called and configuration from environment fails.

    This allows us to import w/o immediate errors so we can actually
    test components in the modules.
    """

    def wrapped(state):
        if config is None:
            raise ImportError(
                f"The {config.name} config could not be initialized due to error: {error}"
            )

        func = getattr(config, func_name)
        return func(state)

    return wrapped
