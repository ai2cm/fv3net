from . import utils
from fv3net.regression import loaders
from vcm import safe
import intake
import yaml
import argparse
from typing import Hashable, Mapping
import logging
import sys

out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(
    logging.Formatter("%(name)s %(asctime)s: %(module)s/L%(lineno)d %(message)s")
)
out_hdlr.setLevel(logging.INFO)
logging.basicConfig(handlers=[out_hdlr], level=logging.INFO)
logger = logging.getLogger("training_data_diags")

GRID_VARS = ["latb", "lonb", "lat", "lon", "area", "land_sea_mask"]

DOMAINS = ["land", "sea", "global"]


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

    return parser.parse_args()


def _open_config(config_path: Hashable) -> Mapping:

    with open(config_path, "r") as f:
        try:
            datasets_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Bad yaml config: {exc}")

    return datasets_config


# start routine

logger.info("Starting diagnostics routine.")

args = _create_arg_parser()

datasets_config = _open_config(args.datasets_config_yml)

# get grid from catalog

cat = intake.open_catalog('catalog.yml')
grid = cat['grid/c48'].to_dask()
print(grid)
grid = grid.drop(labels=['y_interface', "y", 'x_interface', "x"])

# mapping_function = getattr(
#     loaders, datasets_config["one_step_tendencies"]["mapping_function"]
# )
# mapper = mapping_function(datasets_config["one_step_tendencies"]["path"])
# sample_dataset = mapper[list(mapper.keys())[0]]
# grid = (
#     safe.get_variables(sample_dataset, GRID_VARS)
#     .squeeze()
#     .drop(labels=["initial_time", "y", "x", "tile"])
# )

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
    ds_diagnostic = utils.reduce_to_diagnostic(ds_batches, grid, domains=DOMAINS)
    diagnostic_datasets[dataset_name] = ds_diagnostic
    logger.info(f"Finished processing dataset {dataset_name}.")
    

