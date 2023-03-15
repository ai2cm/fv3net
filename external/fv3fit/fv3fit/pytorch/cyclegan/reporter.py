from typing import Mapping, Any


class Reporter:
    """
    Helper class to combine reported metrics to be sent to wandb.
    """

    def __init__(self):
        self.metrics = {}

    def log(self, kwargs: Mapping[str, Any]):
        self.metrics.update(kwargs)

    def clear(self):
        self.metrics.clear()
