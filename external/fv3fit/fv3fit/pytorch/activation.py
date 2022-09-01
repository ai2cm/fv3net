import torch.nn as nn
import dataclasses
from typing import (
    Any,
    Mapping,
)


@dataclasses.dataclass
class ActivationConfig:
    name: str = "ReLU"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def instance(self):
        cls = getattr(nn, self.name)
        return cls(**self.kwargs)
