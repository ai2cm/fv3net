from . import utils
from .config import VARNAMES
from fv3net.regression import loaders
from fv3net.regression.loaders.batch import load_batches
from fv3net.regression.loaders.transform import construct_data_transform
from vcm import safe
import argparse
import yaml
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

GRID_VARS = ['latb', 'lonb', 'lat', 'lon', 'area', 'land_sea_mask']

DOMAINS = ['land', 'sea', 'global']


def _create_arg_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets_config_yml",
        type=str,
        help="Config file with dataset paths, mapping functions, and batch specifications.",
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

mapping_function = getattr(loaders, datasets_config['one_step_tendencies']["mapping_function"])
mapper = mapping_function(datasets_config['one_step_tendencies']["path"])
sample_dataset = mapper[list(mapper.keys())[0]]
grid = (
    safe.get_variables(sample_dataset, GRID_VARS)
    .squeeze()
    .drop(labels=VARNAMES['time_dim'])
)

dataset_names = []
dataset_transforms = {}
for dataset_name, dataset_config in datasets_config.items():
    dataset_names.append(dataset_name)
    mapping_function = getattr(loaders, dataset_config["mapping_function"])
    mapper = mapping_function(dataset_config["path"])
#     sample_dataset = mapper[list(mapper.keys())[0]]
#     grid = (
#         safe.get_variables(sample_dataset, GRID_VARS)
#         .squeeze()
#         .drop(labels=VARNAMES['time_dim'])
#     )
    ds_batches = load_batches(
        mapper,
        construct_data_transform(dataset_transforms),
        dataset_config["variables"],
        **dataset_config["batch_kwargs"],
    )
    ds_diagnostic = utils.reduce_to_diagnostic(ds_batches, grid, domains=DOMAINS)
    print(ds_diagnostic.load())
