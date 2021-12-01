import dataclasses
from typing import Any, Mapping

from ..layers import (
    CombineInputs,
    RNNBlock,
    MLPBlock,
)


def get_architecture_cls(key: str, kwargs: Mapping[str, Any]):
    """
    Grab an instance of an architecture layer block

    Args:
        key: key of block to construct
        kwargs: any keyword arguments for block construction
    """

    if key == "rnn":
        return RNNBlock(**kwargs)
    elif key == "dense":
        return MLPBlock(**kwargs)
    elif key == "linear":
        return MLPBlock(depth=0)
    else:
        raise KeyError(f"Unrecognized architecture provided: {key}")


@dataclasses.dataclass
class ArchitectureConfig:
    """
        name: Name of underlying model architecture to use for the emulator.
            See `get_architecture_cls` for a list of supported layers.
        kwargs: keyword arguments to pass to the initialization
            of the architecture layer
    """

    name: str
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def build(self):
        return get_architecture_cls(self.name, kwargs=self.kwargs)


def get_combine_from_arch_key(key: str):
    """
    Grab an instance of an input combine layer based on
    the core achitecture key.

    Currently focused on special requirements of RNN, while
    everything else just needs inputs concatenated along the
    feature dimension.

    Args:
        key: key of core architecture block being used
    """

    if key == "rnn":
        return CombineInputs(-1, expand_axis=-1)
    else:
        return CombineInputs(-1, expand_axis=None)


def get_outputs_from_arch_key(key: str):
    """
    Grab a hidden layer to output translation layer for
    specific architectures.

    RNN handles outputs special to retain downward dependence
    enforcement.

    Args:
        key: key of core architecture block being used
    """

    if key == "rnn":
        return RNNOutputs()
    else:
        return DenseOutputs()