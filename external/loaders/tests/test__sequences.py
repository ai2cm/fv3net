from functools import partial
from toolz import identity
import xarray as xr
import numpy as np

from loaders.batches._sequences import shuffle

from loaders.batches import Local
from loaders import Map


def _load_func(seq, item):
    return seq[item]


def _load_alarm(seq, item):
    raise ValueError(f"Load called for item: {item}")


def test_FunctinOutputSequence():

    seq = list(range(10))
    fos = Map(partial(_load_func, seq), seq.copy())
    assert len(fos) == 10
    assert fos[0] == 0


def test_FunctinOutputSequence_no_load_on_slice():
    fos = Map(_load_alarm, list(range(10)))
    fos[0:5]


def test_shuffle():
    seq = list(range(150))
    fos = Map(partial(_load_func, seq), seq.copy())
    shuffled = shuffle(fos)
    assert len(shuffled) == len(seq)
    assert tuple(seq) != tuple([item for item in shuffled])


def test_shuffle_no_load_on_shuffle():
    fos = Map(_load_alarm, list(range(10)))
    shuffle(fos)


def test__sequence_map():
    test_seq = Map(identity, [0, 1, 2])
    mul_seq = test_seq.map(lambda x: 2 * x)
    expected = [0, 2, 4]

    assert len(mul_seq) == len(expected)

    for k, val in enumerate(expected):
        assert mul_seq[k] == val


def test__sequence_take():
    input_seq = [0, 1, 2]
    test_seq = Map(identity, [0, 1, 2])
    out_seq = test_seq.take(2)
    expected = input_seq[:2]

    for k, val in enumerate(expected):
        assert out_seq[k] == val


def test__sequence_local(tmpdir):
    ds = xr.Dataset({"a": (["x"], np.array([1]))})
    test_seq = Map(identity, [ds, ds])
    local = test_seq.local(str(tmpdir))

    assert len(local) == len(test_seq)
    for k in range(len(local)):
        xr.testing.assert_equal(local[k], test_seq[k])

    local = Local(str(tmpdir))
    xr.testing.assert_equal(local[0], ds)
