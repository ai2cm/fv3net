import sys
import fsspec
import zarr
import logging
import synth

logging.basicConfig(level=logging.INFO)

# NOTE: this script was used for one-time generation of reference schema
# and is not activaly maintained

url = "gs://vcm-ml-scratch/brianh/one-step-diags-testing/deep-off-testing/hi-res-diags-subset.zarr"  # noqa
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(
    group, coords=("time", "tile", "grid_xt", "grid_yt")
)
synth.dump(schema, sys.stdout)
