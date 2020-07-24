import logging
import os
import argparse
import sys
import tempfile
import yaml
import xarray as xr

from vcm import coarsen
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
)
from vcm.cloud.fsspec import get_fs

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logging.basicConfig(level=logging.INFO)

GRID_SPEC_C384 = "gs://vcm-ml-data/2020-01-06-C384-grid-spec-with-area-dx-dy.zarr"
DIM_RENAME = {
    "grid_xt_coarse": COORD_X_CENTER,
    "grid_yt_coarse": COORD_Y_CENTER,
    "grid_x_coarse": COORD_X_OUTER,
    "grid_y_coarse": COORD_Y_OUTER,
}


def _get_complete_output_path(input_path, output_path):
    if input_path[-1] == "/":
        input_path = input_path[:-1]
    return os.path.join(output_path, os.path.basename(input_path))


def coarsen_c384_diagnostics(args):

    coarsen_diags_config = _get_config(args.config_path)
    output_path = _get_complete_output_path(args.input_path, args.output_path)
    hires_data_vars = coarsen_diags_config["hi-res-data-vars"]
    logging.info(f"Opening C384 diagnostics at: {args.input_path}.")
    diags = _get_remote_diags(args.input_path)
    grid384 = _get_remote_diags(args.grid_spec)
    coarsening_factor = 384 // coarsen_diags_config["target_resolution"]

    # subset variables and rename the dimensions appropriately
    diags384 = diags[hires_data_vars]
    dims_to_rename = {k: v for k, v in DIM_RENAME.items() if k in diags384}
    diags384 = diags384.rename(dims_to_rename)
    logging.info(f"Size of diagnostic data: {diags384.nbytes / 1e9:.2f} GB")

    # coarsen the data
    diags_coarsened = coarsen.weighted_block_average(
        diags384,
        grid384["area"],
        x_dim=COORD_X_CENTER,
        y_dim=COORD_Y_CENTER,
        coarsening_factor=coarsening_factor,
    )
    logging.info(
        f"Done coarsening diagnostics to C{coarsen_diags_config['target_resolution']}."
    )

    diags_coarsened = diags_coarsened.unify_chunks()
    if coarsen_diags_config.get("rechunk") is not None:
        diags_coarsened = diags_coarsened.chunk(coarsen_diags_config["rechunk"])
        logging.info(f"Done rechunking dataset.")
    logging.info(f"Starting to write coarsened diagnostics locally.")
    with tempfile.TemporaryDirectory() as tmpdirname:
        diags_coarsened.to_zarr(tmpdirname, mode="w", consolidated=True)
        logging.info(f"Done writing coarsened diagnostics locally.")
        gsutil.copy(tmpdirname, output_path)
        logging.info(f"Done copy coarsened diagnostics zarr to {output_path}")


def _get_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _get_remote_diags(diags_path):
    fs = get_fs(diags_path)
    mapper = fs.get_mapper(diags_path)
    return xr.open_zarr(mapper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=str, help="GCS location of C384 diagnostics data zarr."
    )
    parser.add_argument(
        "config_path", type=str, help="Location of diagnostics coarsening config yaml."
    )
    parser.add_argument(
        "--grid_spec",
        type=str,
        help=f"GCS location of C384 grid-spec. Defaults to {GRID_SPEC_C384}.",
        default=GRID_SPEC_C384,
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="GCS location where coarsened diagnostics zarrs will be written. "
        "Specifically will be saved at {output_path}/{basename(input_path)},
    )
    args = parser.parse_args()
    coarsen_c384_diagnostics(args)
