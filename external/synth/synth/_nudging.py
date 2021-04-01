import os
from typing import Sequence, Mapping

import numpy as np

from synth.core import DatasetSchema, generate, Range


from .schemas import load_schema_directory


def _generate(
    directory: str, schema: Mapping[str, DatasetSchema], times: Sequence[np.datetime64],
):
    ranges = {"pressure_thickness_of_atmospheric_layer": Range(0.99, 1.01)}
    for relpath, schema in schema.items():
        outpath = os.path.join(directory, relpath)
        (
            generate(schema, ranges)
            .to_zarr(outpath, consolidated=True)
        )


def generate_nudging(outdir: str, times: Sequence[np.datetime64]):
    directory_schema = load_schema_directory("nudge_to_fine")
    _generate(
        outdir, directory_schema, times=times,
    )
