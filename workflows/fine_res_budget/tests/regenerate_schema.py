import logging
import fsspec
import zarr

import synth
import budget.config

logging.basicConfig(level=logging.INFO)


lo_res_coords = ("time", "tile", "grid_xt", "grid_yt", "pfull")

hires_coords = [
    "tile",
    "time",
    "nv",
    "pfull",
    "grid_x_coarse",
    "grid_xt_coarse",
    "grid_y_coarse",
    "grid_yt_coarse",
]


def get_schema(url, coords):

    mapper = fsspec.get_mapper(url)
    group = zarr.open_consolidated(mapper)
    schema = synth.read_schema_from_zarr(group, coords)

    return schema


def save_schema(url, coords, out):
    schema = get_schema(url, coords)
    with open(out, "w") as f:
        synth.dump(schema, f)


save_schema(budget.config.physics_url, hires_coords, "diag.json")
save_schema(budget.config.restart_url, lo_res_coords, "restart.json")
save_schema(
    budget.config.area_url,
    ["tile", "pfull", "grid_xt_coarse", "grid_yt_coarse"],
    "area.json",
)
save_schema(
    budget.config.gfsphysics_url,
    [
        "grid_x_coarse",
        "grid_xt_coarse",
        "grid_y_coarse",
        "grid_yt_coarse",
        "nv",
        "pfull",
        "tile",
        "time",
    ],
    "gfsphysics.json",
)
