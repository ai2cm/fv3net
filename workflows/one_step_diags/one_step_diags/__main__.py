from vcm.cloud import get_protocol
from vcm.cloud.gsutil import copy
from vcm.cubedsphere.constants import TIME_FMT
from vcm.safe import get_variables
from fv3net.pipelines.common import update_nested_dict
from . import utils
from .plots import make_all_plots
from .config import (
    INIT_TIME_DIM,
    FORECAST_TIME_DIM,
    ONE_STEP_ZARR,
    ZARR_STEP_DIM,
    ZARR_STEP_NAMES,
    OUTPUT_NC_FILENAME,
    REPORT_TITLE,
    FIGURE_METADATA_FILE,
    METADATA_TABLE_FILE,
)
from . import config
import report
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import argparse
import xarray as xr
import zarr
import dask
import numpy as np
import fsspec
import yaml
import random
import os
import shutil
from tempfile import NamedTemporaryFile
from typing import Sequence, Mapping, Any
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

dask.config.set(scheduler="single-threaded")

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
        "--report_directory",
        type=str,
        default=None,
        help="(Public) bucket path for report and image upload. If omitted, report is"
        "written to netcdf_output.",
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


def _open_timestamp_pairs(timestamp_pairs: Sequence, mapper: Mapping) -> xr.Dataset:
    """dataflow pipeline func for opening datasets of paired initial times"""

    logger.info(f"Opening the following timesteps: {timestamp_pairs}")

    ds = xr.open_zarr(mapper)
    ds_pair = get_variables(ds, config["VARS_FROM_ZARR"]).sel(
        {
            INIT_TIME_DIM: timestamp_pairs,
            ZARR_STEP_DIM: [
                ZARR_STEP_NAMES["begin"],
                ZARR_STEP_NAMES["after_physics"],
            ],
        }
    )
    logger.info(f"Finished opening the following timesteps: {timestamp_pairs}")

    return ds_pair


def _insert_derived_vars(
    ds: xr.Dataset, hi_res_diags_zarrpath: str, config: Mapping
) -> xr.Dataset:
    """dataflow pipeline func for adding derived variables to the raw dataset
    """

    logger.info(
        f"Inserting derived variables for timestep " f"{ds[INIT_TIME_DIM].values[0]}"
    )

    ds = (
        ds.assign_coords({"z": np.arange(1.0, ds.sizes["z"] + 1.0)})
        .pipe(utils.time_coord_to_datetime)
        .pipe(utils.insert_hi_res_diags, hi_res_diags_zarrpath, config)
        .pipe(utils.insert_derived_vars_from_ds_zarr)
    )

    logger.info(
        f"Finished inserting derived variables for timestep "
        f"{ds[INIT_TIME_DIM].values[0]}"
    )

    return ds


def _insert_states_and_tendencies(
    ds: xr.Dataset, abs_vars: Sequence, level_vars: Sequence
) -> xr.Dataset:
    """dataflow pipeline func for adding states and tendencies
    """

    timestep = ds[INIT_TIME_DIM].values[0].strftime(TIME_FMT)
    logger.info(f"Inserting states and tendencies for timestep {timestep}")
    ds = (
        utils.get_states_and_tendencies(ds)
        .pipe(utils.insert_column_integrated_tendencies)
        .pipe(utils.insert_model_run_differences)
        .pipe(utils.insert_abs_vars, abs_vars)
        .pipe(utils.insert_variable_at_model_level, level_vars)
    )
    ds.attrs[INIT_TIME_DIM] = [ds[INIT_TIME_DIM].item().strftime(TIME_FMT)]
    logger.info(f"Finished inserting states and tendencies for timestep " f"{timestep}")

    return ds


def _insert_means_and_shrink(
    ds: xr.Dataset, grid: xr.Dataset, config: Mapping
) -> xr.Dataset:
    """dataflow pipeline func for adding mean variables and shrinking dataset to
    managable size for aggregation and saving
    """
    timestep = ds[INIT_TIME_DIM].item().strftime(TIME_FMT)
    logger.info(f"Inserting means and shrinking timestep {timestep}")
    ds = ds.merge(grid)
    ds = utils.insert_diurnal_means(ds, config["DIURNAL_VAR_MAPPING"])
    if ds is not None:
        ds = (
            ds.pipe(
                utils.insert_area_means,
                grid["area"],
                list(config["GLOBAL_MEAN_2D_VARS"])
                + list(config["GLOBAL_MEAN_3D_VARS"]),
                ["land_sea_mask", "net_precipitation_physics"],
            )
            .pipe(utils.shrink_ds, config)
            .load()
        )
        logger.info(f"Finished shrinking timestep {timestep}")

    return ds


def _filter_ds(ds: Any):
    """ dataflow pipeline func for filtering out timestamp pairs that failed to
    load due to runtime errors
    """
    if isinstance(ds, xr.Dataset):
        return True
    else:
        logger.warning(f"Excluding timestep that failed due to runtime error.")
        return False


class MeanAndStDevFn(beam.CombineFn):

    """beam CombineFn subclass for taking mean and standard deviation along
    initialization time dimension of aggregated diagnostic dataset
    """

    def __init__(self, std_vars: Sequence = None):
        self.std_vars = std_vars

    def create_accumulator(self):
        return (0.0, 0.0, [], 0)

    def add_input(self, sum_count, input_ds):

        ds = input_ds.drop(INIT_TIME_DIM)
        ds_squared = ds ** 2
        (sum_x, sum_x2, inits, count) = sum_count
        inits = list(set(inits).union(set(ds.attrs[INIT_TIME_DIM])))

        with xr.set_options(keep_attrs=True):
            return sum_x + ds, sum_x2 + ds_squared, inits, count + 1

    def merge_accumulators(self, accumulators):

        sum_xs, sum_x2s, inits_all, counts = zip(*accumulators)
        logger.info(f"Merging accumulations of size: {counts}")

        with xr.set_options(keep_attrs=True):
            return (
                sum(sum_xs),
                sum(sum_x2s),
                sorted([init for inits in inits_all for init in inits]),
                sum(counts),
            )

    def extract_output(self, sum_count):

        logger.info("Computing aggregate means and std. devs.")
        (sum_x, sum_x2, inits, count) = sum_count
        if count:
            with xr.set_options(keep_attrs=True):
                mean = sum_x / count if count else float("NaN")
                mean_of_squares = sum_x2 / count if count else float("NaN")
                std_dev = (
                    np.sqrt(mean_of_squares - mean ** 2)
                    if mean and mean_of_squares
                    else float("NaN")
                )
            for var in mean:
                if var in [f"{var}_global_mean" for var in self.std_vars]:
                    mean = mean.assign({f"{var}_std": std_dev[var]})
                    mean[f"{var}_std"].attrs.update(mean[var].attrs)
                    if "long_name" in mean[var].attrs:
                        mean[f"{var}_std"].attrs.update(
                            {"long_name": mean[var].attrs["long_name"] + "std. dev"}
                        )
            mean = mean.assign_attrs({INIT_TIME_DIM: inits})
        else:
            mean = None

        return mean


def _write_ds(ds: xr.Dataset, fullpath: str):
    """dataflow pipeline func for writing out final netcdf"""

    logger.info(f"Writing final dataset to netcdf at {fullpath}.")
    for var in ds.variables:
        # convert string coordinates unicode strings or else xr.to_netcdf will fail
        if ds[var].dtype == "O":
            ds = ds.assign({var: ds[var].astype("S15").astype("unicode_")})
    ds.attrs[INIT_TIME_DIM] = " ".join(ds.attrs[INIT_TIME_DIM])

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        copy(tmpfile.name, fullpath)


def _write_report(output_report_dir, report_sections, metadata):
    with open(os.path.join(output_report_dir, FIGURE_METADATA_FILE), mode="w") as f:
        yaml.dump(report_sections, f)
    with open(os.path.join(output_report_dir, METADATA_TABLE_FILE), mode="w") as f:
        yaml.dump(metadata, f)
    filename = REPORT_TITLE.replace(" ", "_").replace("-", "_").lower() + ".html"
    html_report = report.create_html(report_sections, REPORT_TITLE, metadata=metadata)
    with open(os.path.join(output_report_dir, filename), "w") as f:
        f.write(html_report)


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

beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
with beam.Pipeline(options=beam_options) as p:
    (
        p
        | "CreateDS" >> beam.Create(timestamp_pairs_subset)
        | "OpenDSPairs" >> beam.Map(_open_timestamp_pairs, mapper)
        | "InsertDerivedVars"
        >> beam.Map(_insert_derived_vars, hi_res_diags_zarrpath, config)
        | "InsertStatesAndTendencies"
        >> beam.Map(
            _insert_states_and_tendencies, config["ABS_VARS"], config["LEVEL_VARS"]
        )
        | "InsertMeanVars" >> beam.Map(_insert_means_and_shrink, grid, config)
        | "FilterDS" >> beam.Filter(_filter_ds)
        | "MeanAndStdDev"
        >> beam.CombineGlobally(MeanAndStDevFn(std_vars=config["GLOBAL_MEAN_2D_VARS"]))
        | "WriteDS" >> beam.Map(_write_ds, output_nc_path)
    )

with fsspec.open(output_nc_path) as ncfile:
    states_and_tendencies = xr.open_dataset(ncfile).load()

# if report output path is GCS location, save results to local output dir first

if args.report_directory:
    report_path = args.report_directory
else:
    report_path = args.netcdf_output

proto = get_protocol(report_path)
if proto == "" or proto == "file":
    output_report_dir = report_path
elif proto == "gs":
    remote_report_path, output_report_dir = os.path.split(report_path.strip("/"))

if os.path.exists(output_report_dir):
    shutil.rmtree(output_report_dir)
os.mkdir(output_report_dir)

logger.info(f"Writing diagnostics plots report to {report_path}")

report_sections = make_all_plots(states_and_tendencies, config, output_report_dir)
metadata = vars(args)
metadata.update(
    {"initializations_processed": states_and_tendencies.attrs[INIT_TIME_DIM]}
)
_write_report(output_report_dir, report_sections, metadata)

# copy report directory to necessary locations
if proto == "gs":
    copy(output_report_dir, remote_report_path)
if args.report_directory:
    copy(output_report_dir, args.netcdf_output)
