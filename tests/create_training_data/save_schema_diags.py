import sys
import fsspec
import zarr
import logging
import synth

logging.basicConfig(level=logging.INFO)


def sample(arr):
    return arr[-1, 0, 0]

url = "gs://vcm-ml-data/testing-noah/2020-04-18/25b5ec1a1b8a9524d2a0211985aa95219747b3c6/coarsen_diagnostics/"
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(group, sample=sample, coords=('time', 'tile', 'grid_xt', 'grid_yt'))
synth.dump(schema, sys.stdout)
