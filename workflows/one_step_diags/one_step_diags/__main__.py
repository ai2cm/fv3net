from vcm.safe import get_variables
from fv3net.pipelines.common import update_nested_dict
from .config import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    ONE_STEP_ZARR,
    ZARR_STEP_DIM,
    OUTPUT_NC_FILENAME,
    CONFIG_FILENAME,
)
from . import config
from .pipeline import run
import argparse
import xarray as xr
import zarr
import fsspec
import yaml
import random
import os
from typing import Sequence, Mapping
import logging
import sys
import warnings

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
out_hdlr.setLevel(logging.INFO)
logging.basicConfig(handlers=[out_hdlr], level=logging.INFO)
logger = logging.getLogger("one_step_diags")

warnings.filterwarnings(
    "ignore",
    message="Dataset.__getitem__ is unsafe. Please avoid use in long-running code.",
)
warnings.filterwarnings(
    "ignore", message="Dataset.stack is unsafe. Please avoid use in long-running code."
)


def _create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "one_step_data", type=str, help="One-step zarr path, not including zarr name."
    )
    parser.add_argument(
        "hi_res_diags",
        type=str,
        help="C384 diagnostics zarr path, not including zarr name.",
    )
    parser.add_argument(
        "timesteps_file",
        type=str,
        help=(
            "File containing paired timesteps for test set. See documentation "
            "in one-steps scripts for more information."
        ),
    )
    parser.add_argument(
        "netcdf_output", type=str, help="Output location for diagnostics netcdf file."
    )
    parser.add_argument(
        "--diags_config",
        type=str,
        default=None,
        help=(
            "File containing one-step diagnostics configuration mapping to guide "
            "plot creation. Plots are specified using configurationn in .config.py"
            " but additional plots can be added by creating entries in the "
            "diags_config yaml."
        ),
    )
    parser.add_argument(
        "--data_fold",
        type=str,
        default=None,
        help=(
            "Whether to use 'train', 'test', or both (None) sets of data in "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--start_ind",
        type=int,
        default=None,
        help="First timestep index to use in "
        "zarr. Earlier spin-up timesteps will be skipped. Defaults to 0.",
    )
    parser.add_argument(
        "--n_sample_inits",
        type=int,
        default=None,
        help="Number of initalizations to use in computing one-step diagnostics.",
    )
    parser.add_argument(
        "--coarsened_diags_zarr_name",
        type=str,
        default="gfsphysics_15min_coarse.zarr",
        help="(Public) bucket path for report and image upload. If omitted, report is"
        "written to netcdf_output.",
    )

    return parser


def _open_diags_config(config_yaml_path: str) -> Mapping:

    with open(config_yaml_path, mode="r") as f:
        diags_config = yaml.safe_load(f)

    return diags_config


def _get_random_subset(timestep_pairs: Sequence, n_samples: int) -> Sequence:

    old_state = random.getstate()
    random.seed(0, version=1)
    timestamp_pairs_subset = random.sample(timestep_pairs, n_samples)
    random.setstate(old_state)

    return timestamp_pairs_subset


# start routine

args, pipeline_args = _create_arg_parser().parse_known_args()

default_config = {key: getattr(config, key) for key in config.__all__}
if args.diags_config is not None:
    supplemental_config = _open_diags_config(args.diags_config)
    config = update_nested_dict(default_config, supplemental_config)
else:
    config = default_config

zarrpath = os.path.join(args.one_step_data, ONE_STEP_ZARR)
mapper = fsspec.get_mapper(zarrpath)
if ".zmetadata" not in mapper:
    logger.info("Consolidating metadata.")
    zarr.consolidate_metadata(mapper)
ds_zarr_template = get_variables(
    xr.open_zarr(mapper), config["GRID_VARS"] + [INIT_TIME_DIM]
)

# get subsampling from json file and specified parameters

with open(args.timesteps_file, "r") as f:
    timesteps = yaml.safe_load(f)
if args.data_fold is not None:
    timestamp_pairs = timesteps[args.data_fold]
else:
    timestamp_pairs = [
        timestep_pair for data_fold in timesteps.values() for timestep_pair in data_fold
    ]

if args.n_sample_inits:
    timestamp_pairs_subset = _get_random_subset(
        timestamp_pairs[(args.start_ind) :], args.n_sample_inits
    )
else:
    timestamp_pairs_subset = timestamp_pairs[(args.start_ind) :]

hi_res_diags_zarrpath = os.path.join(args.hi_res_diags, args.coarsened_diags_zarr_name)

grid = ds_zarr_template.isel(
    {INIT_TIME_DIM: 0, FORECAST_TIME_DIM: 0, ZARR_STEP_DIM: 0}
).drop([ZARR_STEP_DIM, INIT_TIME_DIM, FORECAST_TIME_DIM])

output_nc_path = os.path.join(args.netcdf_output, OUTPUT_NC_FILENAME)

# run the pipeline

run(
    pipeline_args,
    zarrpath,
    timestamp_pairs_subset,
    hi_res_diags_zarrpath,
    config,
    grid,
    output_nc_path,
)

# write config to output location
with fsspec.open(
    os.path.join(args.netcdf_output, CONFIG_FILENAME), "w"
) as config_output_file:
    yaml.dump(config, config_output_file)
