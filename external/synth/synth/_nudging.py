import os
from typing import Sequence

import numpy as np

from synth.core import DatasetSchema, generate

from .schemas import load_schema as _load_schema


def _generate(
    directory: str,
    tendencies_schema: DatasetSchema,
    before_dynamics_schema: DatasetSchema,
    after_dynamics_schema: DatasetSchema,
    after_physics_schema: DatasetSchema,
    after_nudging_schema: DatasetSchema,
    prognostic_diags_schema: DatasetSchema,
    physics_tendency_components_schema: DatasetSchema,
    times: Sequence[np.datetime64],
):
    for relpath, schema in [
        ("before_dynamics.zarr", before_dynamics_schema),
        ("after_dynamics.zarr", after_dynamics_schema),
        ("after_physics.zarr", after_physics_schema),
        ("nudging_tendencies.zarr", tendencies_schema),
        ("after_nudging.zarr", tendencies_schema),
        ("prognostic_diags.zarr", prognostic_diags_schema),
        ("physics_tendency_components.zarr", physics_tendency_components_schema),
    ]:
        outpath = os.path.join(directory, relpath)
        (generate(schema).assign_coords(time=times).to_zarr(outpath, consolidated=True))


def generate_nudging(outdir: str, times: Sequence[np.datetime64]):
    _generate(
        outdir,
        after_dynamics_schema=_load_schema("after_dynamics.json"),
        # I don't think this matters, the schema should be the same
        before_dynamics_schema=_load_schema("after_dynamics.json"),
        after_nudging_schema=_load_schema("after_dynamics.json"),
        after_physics_schema=_load_schema("after_physics.json"),
        tendencies_schema=_load_schema("nudging_tendencies.json"),
        prognostic_diags_schema=_load_schema("prognostic_diags.json"),
        physics_tendency_components_schema=_load_schema(
            "physics_tendency_components.json"
        ),
        times=times,
    )
