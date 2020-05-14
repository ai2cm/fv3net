"""
Example of timesteps_file input data::

    {
        "train": [
            [ "20160801.003000", "20160801.004500" ],
            [ "20160801.001500", "20160801.003000" ]
        ],
        "test": [
            [ "20160801.011500", "20160801.013000" ],
            [ "20160801.010000", "20160801.011500" ]
        ]
    }⏎
"""
import os
import argparse
import logging

import yaml
import fsspec
import xarray as xr
import zarr

from .config import get_config
from .pipeline import run

logger = logging.getLogger(__file__)

ZARR_NAME = "big.zarr"
COARSENED_DIAGS_ZARR_NAME = "gfsphysics_15min_coarse.zarr"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(epilog=__doc__)
    parser.add_argument(
        "--variable_namefile",
        type=str,
        default=None,
        help="yaml file for updating the default data variable names.",
    )
    # TODO would be cleaner to point directly to the big zarr rather
    # than the directory containing it. Then we could remove the ZARR_NAME global.
    parser.add_argument(
        "gcs_input_data_path",
        type=str,
        help="Location of input data in Google Cloud Storage bucket. "
        "Don't include bucket or big.zarr in path.",
    )
    parser.add_argument(
        "diag_c48_path",
        type=str,
        help="Directory containing diagnostic zarr directory coarsened from C384 "
        "for features (SHF, LHF, etc.) that are not saved in restarts.",
    )
    parser.add_argument(
        "timesteps_file",
        type=str,
        help=(
            "File containing paired timesteps for test and/or train sets. See the "
            "__doc__ of this script for more info."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Write path for train data in Google Cloud Storage bucket. "
        "Don't include bucket in path.",
    )

    args, pipeline_args = parser.parse_known_args()
    logging.basicConfig(level=logging.INFO)

    if args.variable_namefile is None:
        updates = {}
    else:
        with open(args.variable_namefile, "r") as f:
            updates = yaml.safe_load(f)
    names = get_config(updates)

    with open(args.timesteps_file, "r") as f:
        timesteps = yaml.safe_load(f)

    big_zarr_path = os.path.join(args.gcs_input_data_path, ZARR_NAME)
    mapper = fsspec.get_mapper(big_zarr_path)
    if ".zmetadata" not in mapper:
        logger.info("Consolidating metadata")
        zarr.consolidate_metadata(mapper)
    ds = xr.open_zarr(mapper, consolidated=True)

    # TODO refactor up to main
    full_zarr_path = os.path.join(args.diag_c48_path, COARSENED_DIAGS_ZARR_NAME)
    mapper = fsspec.get_mapper(full_zarr_path)
    ds_diag = xr.open_zarr(mapper, consolidated=True)

    # TODO Basic io of diag_c48_path should be lifted here as well
    run(ds, ds_diag, args.output_dir, pipeline_args, names, timesteps)
