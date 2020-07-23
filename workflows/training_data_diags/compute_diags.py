import diagnostics_utils as utils
from loaders import batches
from vcm.cloud import get_fs
import xarray as xr
from tempfile import NamedTemporaryFile
import intake
import yaml
import argparse
from typing import Mapping
import sys
import os
import logging
import uuid

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("training_data_diags")

DOMAINS = ["land", "sea", "global"]
OUTPUT_DIAGS_NC_NAME = "diagnostics"
OUTPUT_DIURNAL_NC_NAME = "diurnal_cycle"
TIME_DIM = "time"

DIURNAL_CYCLE_VARS = [
    "column_integrated_dQ1",
    "column_integrated_dQ2",
    "column_integrated_pQ1",
    "column_integrated_pQ2",
    "column_integrated_Q1",
    "column_integrated_Q2",
]


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets_config_yml",
        type=str,
        help=(
            "Config file with dataset paths, mapping functions, and batch"
            "specifications."
        ),
    )
    parser.add_argument(
        "output_path",
        type=str,
        help=("Local or remote path where diagnostic dataset will be written."),
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="Json file that defines train timestep set.",
    )
    return parser.parse_args()


def _open_config(config_path: str) -> Mapping:

    with open(config_path, "r") as f:
        try:
            datasets_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")

    return datasets_config


def _write_nc(ds: xr.Dataset, output_path: str, output_nc_name: str):
    output_file = os.path.join(
        output_path, output_nc_name + "_" + str(uuid.uuid1())[-6:] + ".nc"
    )
    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        get_fs(output_path).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


if __name__ == "__main__":

    logger.info("Starting diagnostics routine.")

    args = _create_arg_parser()

    datasets_config = _open_config(args.datasets_config_yml)

    cat = intake.open_catalog("catalog.yml")
    grid = cat["grid/c48"].to_dask()
    grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])
    surface_type = cat["landseamask/c48"].to_dask()
    surface_type = surface_type.drop(labels=["y", "x"])
    grid = grid.merge(surface_type)

    variable_names = datasets_config["variables"]
    batch_kwargs = datasets_config["batch_kwargs"]

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        datasets_config["batch_kwargs"]["timesteps"] = timesteps

    diagnostic_datasets, diurnal_cycle_datasets = {}, {}

    for dataset_name, dataset_config in datasets_config["sources"].items():
        logger.info(f"Reading dataset {dataset_name}.")
        ds_batches = batches.diagnostic_batches_from_geodata(
            dataset_config["path"],
            variable_names,
            mapping_function=dataset_config["mapping_function"],
            mapping_kwargs=dataset_config.get("mapping_kwargs", None),
            **batch_kwargs,
        )
        batches_diags, batches_diurnal = [], []
        for i, ds in enumerate(ds_batches):
            logger.info(
                f"Computing batch {i+1}/{len(ds_batches)} diagnostics for source {dataset_name}..."
            )
            ds = xr.concat(ds_batches, dim=TIME_DIM)
            ds = (
                ds.pipe(utils.insert_total_apparent_sources)
                .pipe(utils.insert_column_integrated_vars)
                .load()
            )
            ds_batch_diagnostic = utils.reduce_to_diagnostic(ds, grid, domains=DOMAINS)
            batches_diags.append(ds_batch_diagnostic)
            logger.info(
                f"Computing batch {i+1}/{len(ds_batches)} diurnal cycles for source {dataset_name}..."
            )

            ds_batch_diurnal = utils.create_diurnal_cycle_dataset(
                ds,
                longitude=grid["lon"],
                land_sea_mask=grid["land_sea_mask"],
                diurnal_vars=DIURNAL_CYCLE_VARS,
            )
            batches_diurnal.append(ds_batch_diurnal)
        ds_diagnostic = xr.concat(batches_diags, dim="batch").mean("batch")
        ds_diurnal = xr.concat(batches_diurnal, dim="batch").mean("batch")
        diagnostic_datasets[dataset_name] = ds_diagnostic
        diurnal_cycle_datasets[dataset_name] = ds_diurnal
        logger.info(f"Finished processing dataset {dataset_name}.")

    for datasets, output_name in zip(
        [diagnostic_datasets, diurnal_cycle_datasets],
        [OUTPUT_DIAGS_NC_NAME, OUTPUT_DIURNAL_NC_NAME],
    ):
        ds_result = xr.concat(
            [
                dataset.expand_dims({"data_source": [dataset_name]})
                for dataset_name, dataset in datasets.items()
            ],
            dim="data_source",
        )
        _write_nc(ds_result, args.output_path, output_name)
