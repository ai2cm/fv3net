from typing_extensions import Protocol
from typing import Dict


# https://stackoverflow.com/questions/54668000/
class Dataclass(Protocol):
    __dataclass_fields__: Dict
