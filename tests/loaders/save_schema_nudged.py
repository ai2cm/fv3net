import fsspec
import zarr
import synth
import os
from pathlib import Path

script_path = str(Path(__file__).parent)

url = "gs://vcm-ml-scratch/andrep/nudging/2020-05-09-nudge-5day/outdir-3h"

after_physics = os.path.join(url, "after_physics.zarr")
nudge = os.path.join(url, "nudging_tendencies.zarr")


def schema_to_file(data_url, output_filepath):

    mapper = fsspec.get_mapper(data_url)
    group = zarr.open_group(mapper)
    schema = synth.read_schema_from_zarr(group, coords=("time", "tile", "z", "y", "x"))

    with open(output_filepath, "w") as stream:
        synth.dump(schema, stream)


schema_to_file(
    after_physics, os.path.join(script_path, "test_nudged_loader", "after_physics.json")
)
schema_to_file(
    nudge, os.path.join(script_path, "test_nudged_loader", "nudging_tendencies.json")
)
