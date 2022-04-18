from dataclasses import dataclass

from dataclass_serialization import __version__
from typing import Union
from dataclass_serialization.yaml import to_yaml, from_yaml
import dacite.types


def test_version():
    assert __version__ == "0.1.0"


def test_to_yaml():
    @dataclass
    class A:
        a: int = 1

    @dataclass
    class B:
        a: A = A()

    b = B()
    loaded = from_yaml(B, to_yaml(b))
    assert loaded.a.a == b.a.a


def test_to_yaml_union():
    @dataclass
    class A1:
        a: int = 1

    @dataclass
    class A2:
        a: int = 2

    @dataclass
    class B:
        a: Union[A1, A2]

    ans = to_yaml(B(A1()))
    assert "!A1" in ans

    ans = to_yaml(B(A2()))
    assert "!A2" in ans


def test_to_yaml_from_yaml_union():
    @dataclass
    class A1:
        a: int = 1

    @dataclass
    class A2:
        a: int = 2

    @dataclass
    class B:
        a: Union[A1, A2]

    b = B(A1())
    loaded = from_yaml(B, to_yaml(b))
    assert loaded.a.a == b.a.a
    assert isinstance(loaded.a, A1)

    b = B(A2())
    loaded = from_yaml(B, to_yaml(b))
    assert isinstance(loaded.a, A2)


def test_extract_generic():
    u = Union[int, str]
    assert dacite.types.extract_generic(u) == (int, str)


def test_from_yaml_union():
    @dataclass
    class A1:
        a: int = 1

    @dataclass
    class A2:
        a: int = 2

    @dataclass
    class B:
        b: Union[A1, A2]

    yaml_str = """
b: !A2
    a: 1
    """

    loaded = from_yaml(B, yaml_str)
    assert isinstance(loaded.b, A2)

    yaml_str = """
b: !A1
    a: 1
    """

    loaded = from_yaml(B, yaml_str)
    assert isinstance(loaded.b, A1)
