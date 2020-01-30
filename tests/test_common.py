from fv3net.pipelines import common
import xarray as xr
import numpy as np


def test__xarray_chunks():
    dim = 'dim_0'
    a = (xr.DataArray(np.ones(10), dims=[dim]).to_dataset(name='a').chunk({dim: 3}))
    assert a.chunks[dim] == (3, 3, 3, 1)


def test_get_chunks_indices_1d():
    chunks = {'x': (5,5)}
    indices = common._get_chunk_indices(chunks)
    assert indices == [
        {'x': slice(0, 5)},
        {'x': slice(5, 10)}
    ]


def _assert_dict_iterables_set_equal(actual, expected):
    assert len(actual) == len(expected)
    for item in actual:
        assert any(item1 == item for item1 in expected)


def test_get_chunks_indices_2d():
    chunks = {'x': (5,5), 'y': (1, 2)}
    indices = common._get_chunk_indices(chunks)
    expected_indices = [
        {'x': slice(0, 5), 'y': slice(0, 1)},
        {'x': slice(0, 5), 'y': slice(1, 3)},
        {'x': slice(5, 10), 'y': slice(0, 1)},
        {'x': slice(5, 10), 'y': slice(1, 3)},
    ]

    _assert_dict_iterables_set_equal(indices, expected_indices)


def test_yield_chunks():
    dim = 'x'
    n = 10
    coords = {dim: np.arange(n)}
    a = xr.Dataset({'a': ([dim], np.ones(10))}, coords=coords).chunk({dim: 3})

    # make sure the length is correct
    assert len(list(common._yield_chunks(a))) == len(a.chunks[dim])

    # check that combining works
    combined = xr.combine_by_coords(chunk.data for chunk in common._yield_chunks(
        a)).sortby(
        dim)
    xr.testing.assert_equal(combined, a)

    # check all items of chunks
    for chunk in common._yield_chunks(a):
        assert isinstance(chunk.data, xr.Dataset)
        assert isinstance(chunk.id, int)
        xr.testing.assert_equal(chunk.data, a.isel(chunk.info[chunk.id]))
        # make sure it exists
        chunk.index





