import pytest
import numpy as np
import tensorflow as tf

from fv3fit.emulation.layers.architecture import CombineInputs
from fv3fit.emulation.models._core import (
    get_architecture_cls,
    get_combine_from_arch_key,
    ArchitectureConfig,
)


def _get_data(shape):

    num = int(np.prod(shape))
    return np.arange(num).reshape(shape).astype(np.float32)


def _get_tensor(shape):
    return tf.convert_to_tensor(_get_data(shape))


def test_get_architecture_unrecognized():

    with pytest.raises(KeyError):
        get_architecture_cls("not_an_arch", {})


@pytest.mark.parametrize(
    "arch_key, expected",
    [
        ("rnn-v1", (20, 4, 3)),
        ("rnn", (20, 4, 3)),
        ("dense", (20, 12)),
        ("linear", (20, 12)),
    ],
)
def test_get_combine_from_arch_key(arch_key, expected):

    combiner = get_combine_from_arch_key(arch_key)
    assert isinstance(combiner, CombineInputs)

    tensor = _get_tensor((20, 4))
    combined = combiner((tensor, tensor, tensor))

    assert combined.shape == expected


def test_ArchParams():
    arch = ArchitectureConfig(name="dense", kwargs=dict(width=128))
    assert isinstance(arch.build(), tf.keras.layers.Layer)


def test_ArchParams_bad_kwargs():
    arch = ArchitectureConfig(name="dense", kwargs=dict(not_a_kwarg="hi"))
    with pytest.raises(TypeError):
        arch.build()
