import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
import pytest
import xarray as xr
import numpy as np
from fv3net.pipelines.common import (
    ChunkXarray,
    _chunk_indices
)


def _dataset(_=None):
    return xr.Dataset({
        'a': (['x'], np.ones(10))
    }).chunk({'x': 3})

class _Dataset(beam.PTransform):
    def expand(self, pcoll):
        return pcoll | beam.Create([None]) | beam.Map(_dataset) 


def test__chunk_indices():
    ds = _dataset()
    output = list(_chunk_indices(ds, ['x']))
    assert len(output) == 4

def test_ChunkXarray():

    def _assert_single_chunk(_, ds: xr.Dataset):
        for dim, chunks in ds.chunks.items():
            if any(chunk != ds.sizes[dim] for chunk in chunks):
                pytest.fail()

    def _assert_key(key, _):
        assert list(key.values()) == ['x']

    with TestPipeline() as p:
        chunks = p | _Dataset() | ChunkXarray(['x'])
        _ = chunks | beam.MapTuple(_assert_single_chunk)
