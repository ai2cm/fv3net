import os
from typing import Sequence

import numpy as np

from synth.core import DatasetSchema, generate

from .schemas import load_schema as _load_schema


def _generate(
    directory: str,
    tendencies_schema: DatasetSchema,
    after_dynamics_schema: DatasetSchema,
    after_physics_schema: DatasetSchema,
    times: Sequence[np.datetime64],
):
    nudging_dir = directory
    nudging_after_dynamics_zarrpath = os.path.join(nudging_dir, "after_dynamics.zarr")
    nudging_after_dynamics_dataset = generate(after_dynamics_schema).assign_coords(
        {"time": times}
    )
    nudging_after_dynamics_dataset.to_zarr(
        nudging_after_dynamics_zarrpath, consolidated=True
    )

    nudging_after_physics_zarrpath = os.path.join(nudging_dir, "after_physics.zarr")
    nudging_after_physics_dataset = generate(after_physics_schema).assign_coords(
        {"time": times}
    )
    nudging_after_physics_dataset.to_zarr(
        nudging_after_physics_zarrpath, consolidated=True
    )

    nudging_tendencies_zarrpath = os.path.join(nudging_dir, "nudging_tendencies.zarr")
    nudging_tendencies_dataset = generate(tendencies_schema).assign_coords(
        {"time": times}
    )
    nudging_tendencies_dataset.to_zarr(nudging_tendencies_zarrpath, consolidated=True)


def generate_nudging(outdir: str, times: Sequence[np.datetime64]):
    _generate(
        outdir,
        after_dynamics_schema=_load_schema("after_dynamics.json"),
        after_physics_schema=_load_schema("after_physics.json"),
        tendencies_schema=_load_schema("nudging_tendencies.json"),
        times=times,
    )
