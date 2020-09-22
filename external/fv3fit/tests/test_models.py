import time
from tensorflow.keras.utils import Sequence

from fv3fit.keras._models.models import _ThreadedSequencePreLoader


class DelayedSequence(Sequence):
    def __init__(self, sequence):
        self.seq = sequence

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        # this must lock in weird way because it takes 60s for 1s sleep
        time.sleep(0.01)
        return self.seq[index]


def test__ThreadedSequencePreLoader():
    sequence = DelayedSequence([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    loader = _ThreadedSequencePreLoader(sequence, num_workers=4)
    result = [item for item in loader]
    assert len(result) == len(sequence)
    for item in result:
        assert item in sequence
