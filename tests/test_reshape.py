from fv3net.regression.dataset_handler import _chunk_indices


def test__chunk_indices():
    chunks = (2, 3)
    expected = [[0, 1], [2, 3, 4]]
    ans = _chunk_indices(chunks)
    assert ans == expected
