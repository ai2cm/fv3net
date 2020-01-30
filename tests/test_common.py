from fv3net.pipelines import common
import xarray as xr
import numpy as np
from toolz.curried import dissoc
from toolz import merge_with

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline


def test__xarray_chunks():
    dim = "dim_0"
    a = xr.DataArray(np.ones(10), dims=[dim]).to_dataset(name="a").chunk({dim: 3})
    assert a.chunks[dim] == (3, 3, 3, 1)


def test_get_chunks_indices_1d():
    chunks = {"x": (5, 5)}
    indices = common._get_chunk_indices(chunks)
    assert indices == [((0,), {"x": slice(0, 5)}), ((1,), {"x": slice(5, 10)})]


def _assert_dict_iterables_set_equal(actual, expected):
    assert len(actual) == len(expected)
    for item in actual:
        assert any(item1 == item for item1 in expected)


def test_get_chunks_indices_2d():
    chunks = {"x": (5, 5), "y": (1, 2)}
    indices = common._get_chunk_indices(chunks)
    expected_indices = [
        ((0, 0), {"x": slice(0, 5), "y": slice(0, 1)}),
        ((0, 1), {"x": slice(0, 5), "y": slice(1, 3)}),
        ((1, 0), {"x": slice(5, 10), "y": slice(0, 1)}),
        ((1, 1), {"x": slice(5, 10), "y": slice(1, 3)}),
    ]

    _assert_dict_iterables_set_equal(indices, expected_indices)


def test_yield_chunks():
    dim = "x"
    n = 10
    coords = {dim: np.arange(n)}
    a = xr.Dataset({"a": ([dim], np.ones(10))}, coords=coords).chunk({dim: 3})

    # make sure the length is correct
    assert len(list(common._yield_chunks(a))) == len(a.chunks[dim])

    # check that combining works
    combined = xr.combine_by_coords(data for _, data in common._yield_chunks(a)).sortby(
        dim
    )
    xr.testing.assert_equal(combined, a)


import uuid


def sum_tuples(sizes):
    return tuple(sum(chunks) for chunks in zip(*sizes))


def _total_size(keys):
    sizes = [key['size'] for key in keys]
    return merge_with(sum_tuples, sizes)


def entrypoint(keys):
    total_size = _total_size(keys)
    print(keys)


def test_open_chunk_zarr(tmpdir):
    tmpdir.chdir()
    dim = "x"
    n = 10
    coords = {dim: np.arange(n)}
    a = xr.Dataset({"a": ([dim], np.ones(10))}, coords=coords).chunk({dim: 3})

    with TestPipeline() as p:
        data = (
            p
            | beam.Create([a])
            | common.SplitChunks()
            | "RandomKey"
            >> beam.Map(
                lambda x: {
                    "chunk": x[0],
                    "id": str(uuid.uuid4()),
                    "size": {name: x[1][name].data.shape for name in x[1]},
                    "ndim": {name: x[1][name].data.ndim for name in x[1]},
                    "data": x[1],
                }
            )
        )

        keys = (
            data
            | beam.combiners.ToList()
            | beam.Map(entrypoint)
        )
        write_to_disk = data | beam.Map(lambda x: x["data"].to_zarr(x["id"]))
