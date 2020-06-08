import logging

import fsspec
import zarr

import synth

logging.basicConfig(level=logging.INFO)


diagurl = "gs://vcm-ml-raw/2020-05-27-40-day-X-SHiELD-simulation-C384-diagnostics/atmos_15min_coarse_ave.zarr"  # noqa
restart_url = "gs://vcm-ml-experiments/2020-06-02-fine-res/2020-05-27-40-day-X-SHiELD-simulation-C384-restart-files.zarr"  # noqa

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
    group = zarr.open_group(mapper)
    schema = synth.read_schema_from_zarr(group, coords)

    return schema


dschema = get_schema(diagurl, hires_coords)
rschema = get_schema(restart_url, lo_res_coords)

with open("diag.json", "w") as f:
    synth.dump(dschema, f)

with open("restart.json", "w") as f:
    synth.dump(rschema, f)
