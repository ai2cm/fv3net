import logging

import fsspec
import zarr

import synth

from typing import Tuple

logging.basicConfig(level=logging.INFO)


REFERENCE_DATASETS = {
    "restart": "gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr",  # noqa
    "atmos_15min_coarse_ave": "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr",  # noqa
    "gfsphysics_15min_coarse": "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/gfsphysics_15min_coarse.zarr",  # noqa
}
COORDS = {
    "restart": ("time", "tile", "grid_xt", "grid_yt", "pfull"),
    "atmos_15min_coarse_ave": (
        "tile",
        "time",
        "nv",
        "pfull",
        "grid_x_coarse",
        "grid_xt_coarse",
        "grid_y_coarse",
        "grid_yt_coarse",
    ),
    "gfsphysics_15min_coarse": (
        "tile",
        "time",
        "pfull",
        "grid_x_coarse",
        "grid_xt_coarse",
        "grid_y_coarse",
        "grid_yt_coarse",
    ),
}


def get_schema(url: str, coords: Tuple[str]) -> synth.DatasetSchema:
    mapper = fsspec.get_mapper(url)
    group = zarr.open_group(mapper)
    schema = synth.read_schema_from_zarr(group, coords)
    return schema


def write_schema(schema: synth.DatasetSchema, name: str):
    with open(f"{name}_schema.json", "w") as file:
        synth.dump(schema, file)


for name, url in REFERENCE_DATASETS.items():
    schema = get_schema(url, COORDS[name])
    write_schema(schema, name)
