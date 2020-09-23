from fv3fit.keras._models.models import _ThreadedSequencePreLoader


def test__ThreadedSequencePreLoader():
    """ Check correctness of the pre-loaded sequence"""
    sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    loader = _ThreadedSequencePreLoader(sequence, num_workers=4)
    result = [item for item in loader]
    assert len(result) == len(sequence)
    for item in result:
        assert item in sequence
