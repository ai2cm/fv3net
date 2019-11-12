from fv3net.models.reshape import *


def test_chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = chunk_indices(chunks)
    assert ans == expected
