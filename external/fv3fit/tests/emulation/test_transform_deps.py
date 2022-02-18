from dataclasses import dataclass
from typing import Set

from fv3fit.emulation.transforms.factories import TransformFactory, _get_dependencies


@dataclass
class DummyFactory(TransformFactory):

    to: str
    deps: Set[str]

    @property
    def required_names(self) -> Set[str]:
        return self.deps


def test_cycle():

    factories = [DummyFactory("b", deps={"a"}), DummyFactory("a", deps={"b"})]

    result = _get_dependencies("a", factories)
    assert result == ({"a"}, {"b"})


def test_intermediate_blend():

    factories = [
        DummyFactory("c", deps={"a", "b"}),
        DummyFactory("d", deps={"b", "c"}),
    ]

    result = _get_dependencies("d", factories)
    assert result == ({"a", "b"}, {"c"})


def test_cycle_with_passthrough():

    factories = [
        DummyFactory("b", deps={"a"}),
        DummyFactory("a", deps={"b"}),
        DummyFactory("a", deps={"a"}),
    ]

    result = _get_dependencies("a", factories)
    assert result == ({"a"}, {"b"})


def test_only_get_requested():

    factories = [
        DummyFactory("b", deps={"a"}),
        DummyFactory("d", deps={"c"}),
    ]

    result = _get_dependencies("d", factories)
    assert result == ({"c"}, set())
