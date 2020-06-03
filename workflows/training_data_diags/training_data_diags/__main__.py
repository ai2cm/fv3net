from . import utils
from .config import VARNAMES
from fv3net.regression import loaders
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

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
out_hdlr.setLevel(logging.INFO)
logging.basicConfig(handlers=[out_hdlr], level=logging.INFO)
logger = logging.getLogger("training_data_diags")

DOMAINS = ["land", "sea", "global"]
OUTPUT_NC_NAME = "diagnostics"


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

    return parser.parse_args()


def _open_config(config_path: str) -> Mapping:

    with open(config_path, "r") as f:
        try:
            datasets_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")

    return datasets_config


def _write_nc(ds: xr.Dataset, output_path: str):
    output_file = os.path.join(
        output_path, OUTPUT_NC_NAME + "_" + str(uuid.uuid1())[-6:] + ".nc"
    )
    with NamedTemporaryFile() as tmpfile:
        ds.to_netcdf(tmpfile.name)
        get_fs(output_path).put(tmpfile.name, output_file)
    logger.info(f"Writing netcdf to {output_file}")


# start routine

logger.info("Starting diagnostics routine.")

args = _create_arg_parser()

datasets_config = _open_config(args.datasets_config_yml)

# get grid from catalog

cat = intake.open_catalog("catalog.yml")
grid = cat["grid/c48"].to_dask()
grid = grid.drop(labels=["y_interface", "y", "x_interface", "x"])

diagnostic_datasets = {}
for dataset_name, dataset_config in datasets_config.items():
    logger.info(f"Reading dataset {dataset_name}.")
    mapping_function = getattr(loaders, dataset_config["mapping_function"])
    mapper = mapping_function(dataset_config["path"])
    ds_batches = loaders.mapper_to_diagnostic_sequence(
        mapper,
        dataset_config["variables"],
        rename_variables=dataset_config.get("rename_variables", None),
        **dataset_config["batch_kwargs"],
    )
    if dataset_name == "one_step_tendencies":
        # cache the land_sea_mask from the one-step data since that variable
        # is missing from the fine-res budget and grid datasets
        surface_type = (
            ds_batches[0][VARNAMES["surface_type_var"]]
            .squeeze()
            .drop(labels=VARNAMES["time_dim"])
        )
        grid = grid.assign({VARNAMES["surface_type_var"]: surface_type})
    ds_diagnostic = utils.reduce_to_diagnostic(ds_batches, grid, domains=DOMAINS)
    if dataset_name == "one_step_tendencies":
        ds_diagnostic = ds_diagnostic.drop(VARNAMES["surface_type_var"])
    diagnostic_datasets[dataset_name] = ds_diagnostic
    logger.info(f"Finished processing dataset {dataset_name}.")

diagnostics_all = xr.concat(
    [
        dataset.expand_dims({"data_source": [dataset_name]})
        for dataset_name, dataset in diagnostic_datasets.items()
    ],
    dim="data_source",
).load()

_write_nc(diagnostics_all, args.output_path)
