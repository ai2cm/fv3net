from . import utils
from .config import INIT_TIME_DIM, ZARR_STEP_DIM, ZARR_STEP_NAMES
from vcm.cloud.gsutil import copy
from vcm.safe import get_variables
from vcm.cubedsphere.constants import TIME_FMT
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import xarray as xr
import numpy as np
import dask
import fsspec
from tempfile import NamedTemporaryFile
from typing import Mapping, Sequence, Any
import logging

logger = logging.getLogger(__name__)

dask.config.set(scheduler="single-threaded")


def run(
    pipeline_args: Sequence,
    zarrpath: str,
    timestep_pairs: Sequence,
    hi_res_diags_zarrpath: str,
    config: Mapping,
    grid: xr.Dataset,
    output_nc_path: str,
):

    beam_options = PipelineOptions(flags=pipeline_args, save_main_session=True)
    with beam.Pipeline(options=beam_options) as p:
        (
            p
            | "CreateDS" >> beam.Create(timestep_pairs)
            | "OpenDSPairs" >> beam.Map(_open_timestamp_pairs, zarrpath, config)
            | "InsertDerivedVars"
            >> beam.Map(_insert_derived_vars, hi_res_diags_zarrpath, config)
            | "InsertStatesAndTendencies"
            >> beam.Map(
                _insert_states_and_tendencies, config["ABS_VARS"], config["LEVEL_VARS"]
            )
            | "InsertMeanVars" >> beam.Map(_insert_means_and_shrink, grid, config)
            | "FilterDS" >> beam.Filter(_filter_ds)
            | "MeanAndStdDev"
            >> beam.CombineGlobally(
                MeanAndStDevFn(std_vars=config["GLOBAL_MEAN_2D_VARS"])
            )
            | "WriteDS" >> beam.Map(_write_ds, output_nc_path)
        )


def _open_timestamp_pairs(
    timestamp_pairs: Sequence, zarrpath: str, config: Mapping
) -> xr.Dataset:
    """dataflow pipeline func for opening datasets of paired initial times"""

    logger.info(f"Opening the following timesteps: {timestamp_pairs}")

    ds = xr.open_zarr(fsspec.get_mapper(zarrpath))
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
        # enforce dimension order for 2-D vars
        spatial_dims = ["tile", "y", "x"]
        if "x" in ds[var].dims:
            non_spatial_dims = [dim for dim in ds[var].dims if dim not in spatial_dims]
            transpose_dims = non_spatial_dims + spatial_dims
            ds[var] = ds[var].transpose(*transpose_dims)

    ds.attrs[INIT_TIME_DIM] = " ".join(ds.attrs[INIT_TIME_DIM])

    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        copy(tmpfile.name, fullpath)
