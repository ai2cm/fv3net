import sys
import fsspec
import logging
import synth

logging.basicConfig(level=logging.INFO)


# url = "gs://vcm-ml-data/test-end-to-end-integration/integration-debug-4c4c163556d0/one_step_run/big.zarr"  # noqa
url = "gs://vcm-ml-scratch/brianh/one-step-diags-testing/physics-off-80a8e4" # noqa
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(group)
synth.dump(schema, sys.stdout)
