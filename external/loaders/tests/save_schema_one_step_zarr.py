import sys
import fsspec
import zarr
import logging
import synth

logging.basicConfig(level=logging.INFO)

# NOTE: this script was used for one-time generation of reference schema
# and is not maintained

url = "gs://vcm-ml-scratch/annak/temp_data/test/20160907.003000.zarr"
mapper = fsspec.get_mapper(url)
group = zarr.open_group(mapper)
schema = synth.read_schema_from_zarr(
    group, coords=("initial_time", "tile", "x", "y", "x_interface", "y_interface")
)
synth.dump(schema, sys.stdout)
