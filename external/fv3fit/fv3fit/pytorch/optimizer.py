import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import dataclasses
from typing import (
    Any,
    Mapping,
    Optional,
    Protocol,
)


@dataclasses.dataclass
class OptimizerConfig:
    name: str = "AdamW"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def instance(self, params):
        cls = getattr(optim, self.name)

        kwargs = dict(params=params, **self.kwargs)

        return cls(**kwargs)


class Scheduler(Protocol):
    def step(self):
        ...


class NullScheduler:
    def step(self):
        pass


@dataclasses.dataclass
class SchedulerConfig:
    name: Optional[str] = None
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def instance(self, optimizer) -> Scheduler:
        if self.name is None:
            return NullScheduler()
        else:
            cls = getattr(lr_scheduler, self.name)

            return cls(optimizer, **self.kwargs)
