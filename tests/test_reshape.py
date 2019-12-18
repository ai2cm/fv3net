from fv3net.regression.reshape import chunk_indices


def test_chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = chunk_indices(chunks)
    assert ans == expected
