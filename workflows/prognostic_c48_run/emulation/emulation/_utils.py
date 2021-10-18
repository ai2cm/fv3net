
from typing import Callable


def wrap_configurable_hook(config, error, func_name) -> Callable:
    """
    This function wraps those call-py-fort hooks to fail when the
    functions are called and environment configuration fails.

    Used because environment variable-based configuration when
    using call-py-fort is hard to test.  This allows us to
    import w/o immediate errors.
    """

    def wrapped(state):
        if config is None:
            raise ImportError(
                f"Monitor store could not be initialized due to error: {error}"
            )

        func = getattr(config, func_name)
        return func(state)

    return wrapped
