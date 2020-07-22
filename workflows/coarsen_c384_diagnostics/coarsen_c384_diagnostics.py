import logging
import os
import shutil
import argparse
import sys
import yaml
import xarray as xr

from vcm import coarsen
from vcm.cloud import gsutil
from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
)
from vcm.cloud.fsspec import get_fs
from fv3net import COARSENED_DIAGS_ZARR_NAME

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logging.basicConfig(level=logging.INFO)


def coarsen_c384_diagnostics(args):

    coarsen_diags_config = _get_config(args.config_path)
    output_path = os.path.join(args.output_path, COARSENED_DIAGS_ZARR_NAME)
    hires_data_vars = coarsen_diags_config["hi-res-data-vars"]
    logging.info(f"Opening C384 diagnostics at: {args.input_path}.")
    diags = _get_remote_diags(args.input_path)
    logging.info(f"Size of diagnostic data: {diags.nbytes / 1e9:.2f} GB")
    coarsening_factor = 384 // coarsen_diags_config["target_resolution"]

    # rename the dimensions appropriately
    grid384 = diags[
        [
            "grid_lat_coarse",
            "grid_latt_coarse",
            "grid_lon_coarse",
            "grid_lont_coarse",
            "area_coarse",
        ]
    ]
    diags384 = xr.merge([diags[hires_data_vars], grid384]).rename(
        {
            "grid_lat_coarse": VAR_LAT_OUTER,
            "grid_latt_coarse": VAR_LAT_CENTER,
            "grid_lon_coarse": VAR_LON_OUTER,
            "grid_lont_coarse": VAR_LON_CENTER,
            "grid_xt_coarse": COORD_X_CENTER,
            "grid_yt_coarse": COORD_Y_CENTER,
            "grid_x_coarse": COORD_X_OUTER,
            "grid_y_coarse": COORD_Y_OUTER,
        }
    )

    # coarsen the data
    diags_coarsened = coarsen.weighted_block_average(
        diags384[hires_data_vars],
        diags384["area_coarse"],
        x_dim=COORD_X_CENTER,
        y_dim=COORD_Y_CENTER,
        coarsening_factor=coarsening_factor,
    )
    logging.info(f"Done coarsening diagnostics to C{coarsen_diags_config['target_resolution']}.")

    diags_coarsened = diags_coarsened.unify_chunks()
    diags_coarsened = diags_coarsened.chunk(coarsen_diags_config['rechunk'])
    logging.info(f"Done rechunking dataset.")
#     diags_coarsened = diags_coarsened.load()
#     logging.info(f"Done loading dataset.")
    diags_coarsened.to_zarr(COARSENED_DIAGS_ZARR_NAME, mode="w", consolidated=True)
    logging.info(f"Done writing coarsened diagnostics locally.")
    gsutil.copy(COARSENED_DIAGS_ZARR_NAME, output_path)
    logging.info(f"Done copy coarsened diagnostics zarr to {output_path}")
    shutil.rmtree(COARSENED_DIAGS_ZARR_NAME)


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
        "input_path", type=str, help="GCS location of C384 diagnostics data zarrs."
    )
    parser.add_argument(
        "config_path", type=str, help="Location of diagnostics coarsening config yaml."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="GCS location where= coarsened diagnostics zarrs will be written.",
    )
    args = parser.parse_args()
    coarsen_c384_diagnostics(args)
