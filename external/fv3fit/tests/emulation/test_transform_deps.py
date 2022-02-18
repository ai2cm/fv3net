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
    assert result == {"a"}


def test_intermediate():

    factories = [
        DummyFactory("c", deps={"a", "b"}),
        DummyFactory("d", deps={"b", "c"}),
    ]

    result = _get_dependencies("d", factories)
    assert result == {"a", "b"}


def test_only_requested():

    factories = [
        DummyFactory("b", deps={"a"}),
        DummyFactory("d", deps={"c"}),
    ]

    result = _get_dependencies("d", factories)
    assert result == {"c"}
