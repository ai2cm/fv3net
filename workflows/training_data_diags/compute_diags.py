import diagnostics_utils as utils
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

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
handler.setLevel(logging.INFO)
logging.basicConfig(handlers=[handler], level=logging.INFO)
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

    diagnostic_datasets = {}
    for dataset_name, dataset_config in datasets_config["sources"].items():
        logger.info(f"Reading dataset {dataset_name}.")
        batch_function = getattr(loaders, dataset_config["batch_function"])
        ds_batches = loaders.diagnostic_sequence_from_mapper(
            dataset_config["path"],
            variable_names,
            rename_variables=dataset_config.get("rename_variables", None),
            **dataset_config["batch_kwargs"],
        )
        ds_diagnostic = utils.reduce_to_diagnostic(ds_batches, grid, domains=DOMAINS)
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
