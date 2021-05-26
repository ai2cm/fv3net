from typing_extensions import Protocol
from typing import Dict


class Dataclass(Protocol):
    __dataclass_fields__: Dict
