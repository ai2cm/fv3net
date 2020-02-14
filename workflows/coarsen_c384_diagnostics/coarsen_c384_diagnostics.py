import intake
import logging
import os
import shutil
import argparse
import yaml
import fsspec
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
from vcm.fv3_restarts import _split_url

logging.basicConfig(level=logging.INFO)


def coarsen_c384_diagnostics(args):
    
    coarsen_diags_config = _get_config(args.config_path)
    zarr_suffix = coarsen_diags_config["output_filename"]
    output_path = os.path.join(args.output_path, zarr_suffix)
    hires_data_vars = coarsen_diags_config["hi-res-data-vars"]
    diags = _get_remote_diags(args.input_path)
    logging.info(f"Size of diagnostic data:  {diags.nbytes / 1e9:.2f} GB")
    coarsening_factor = 384//coarsen_diags_config['target_resolution']

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

    diags_coarsened = diags_coarsened.unify_chunks()
    diags_coarsened.to_zarr(zarr_suffix, mode="w", consolidated=True)
    gsutil.copy(zarr_suffix, output_path)
    logging.info(f"Done writing coarsened diagnostics zarr to {output_path}")
    shutil.rmtree(zarr_suffix)
    
    
def _get_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
def _get_remote_diags(diags_path):
    proto, path = _split_url(diags_path)
    fs = fsspec.filesystem(proto)
    mapper = fs.get_mapper(diags_path)
    return xr.open_zarr(mapper)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="GCS location of C384 diagnostics data zarrs."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="GCS location where= coarsened diagnostics zarrs will be written."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Location of diagnostics coarsening config yaml."
    )
    args = parser.parse_args()
    coarsen_c384_diagnostics(args)
    
    