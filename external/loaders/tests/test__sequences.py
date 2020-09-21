from functools import partial

from loaders.batches._sequences import FunctionOutputSequence, shuffle


def _load_func(seq, item):
    return seq[item]


def _load_alarm(seq, item):
    raise ValueError(f"Load called for item: {item}")


def test_FunctinOutputSequence():

    seq = list(range(10))
    fos = FunctionOutputSequence(partial(_load_func, seq), seq.copy())
    assert len(fos) == 10
    assert fos[0] == 0


def test_FunctinOutputSequence_no_load_on_slice():
    fos = FunctionOutputSequence(_load_alarm, list(range(10)))
    fos[0:5]


def test_shuffle():
    seq = list(range(150))
    fos = FunctionOutputSequence(partial(_load_func, seq), seq.copy())
    shuffled = shuffle(fos)
    assert len(shuffled) == len(seq)
    assert tuple(seq) != tuple([item for item in shuffled])


def test_shuffle_no_load_on_shuffle():
    fos = FunctionOutputSequence(_load_alarm, list(range(10)))
    shuffle(fos)
