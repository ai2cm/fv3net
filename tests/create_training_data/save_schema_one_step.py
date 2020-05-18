import sys
import fsspec
import zarr
import logging
import synth

logging.basicConfig(level=logging.INFO)


def sample(arr):
    return arr[-1, 0, 0, 0]


url = "gs://vcm-ml-data/test-end-to-end-integration/integration-debug-4c4c163556d0/one_step_run/big.zarr"  # noqa
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(group, sample=sample)
synth.dump(schema, sys.stdout)
