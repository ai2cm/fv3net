from dataclasses import asdict
from enum import Enum


def asdict_with_enum(obj):
    """Recursively turn a dataclass obj into a dictionary handling any Enums
    """

    def _generate(x):
        for key, val in x:
            if isinstance(val, Enum):
                yield key, val.value
            else:
                yield key, val

    def dict_factory(x):
        return dict(_generate(x))

    return asdict(obj, dict_factory=dict_factory)
