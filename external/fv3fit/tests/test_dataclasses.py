from dataclasses import dataclass
from enum import Enum
from fv3fit.dataclasses import asdict_with_enum


def test_asdict_with_enum():
    class A(Enum):
        a = 1

    @dataclass
    class B:
        enum: A

    assert asdict_with_enum(B(A.a)) == {"enum": 1}
