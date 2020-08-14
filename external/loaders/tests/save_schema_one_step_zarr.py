import sys
import fsspec
import zarr
import logging
import synth

logging.basicConfig(level=logging.INFO)

# NOTE: this script was used for one-time generation of reference schema
# and is not maintained

url = "gs://vcm-ml-scratch/annak/2020-08-12-hybrid-nudge-to-obs/initial-run/prognostic_run/data.zarr/"
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(
    group, coords=("time", "tile", "z", "y", "x")
)
synth.dump(schema, sys.stdout)
