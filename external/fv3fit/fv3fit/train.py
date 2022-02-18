import argparse
import logging
import os
from typing import Optional, Tuple
from fv3fit._shared.config import get_arg_updated_config_dict
import yaml
import dataclasses
import fsspec

import fv3fit.keras
import fv3fit.sklearn
import fv3fit
import loaders
import loaders.typing
import tempfile
from loaders.batches.save import main as save_main

from vcm.cloud import copy

logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_config", type=str, help="path of fv3fit.TrainingConfig yaml file",
    )
    parser.add_argument(
        "training_data_config",
        type=str,
        help="path of loaders.BatchesLoader training data yaml file",
    )
    parser.add_argument(
        "output_path", type=str, help="path to save config and trained model"
    )
    parser.add_argument(
        "--validation-data-config",
        type=str,
        default=None,
        help=(
            "path of loaders.BatchesLoader validation data yaml file, "
            "by default an empty sequence is used"
        ),
    )
    parser.add_argument(
        "--local-download-path",
        type=str,
        default=None,
        help=("path for downloading data before training"),
    )
    return parser


def dump_dataclass(obj, yaml_filename):
    with fsspec.open(yaml_filename, "w") as f:
        yaml.safe_dump(dataclasses.asdict(obj), f)


def get_data(
    training_data_config: str,
    validation_data_config: Optional[str],
    local_download_path: Optional[str],
) -> Tuple[loaders.typing.Batches, loaders.typing.Batches]:
    if local_download_path is None:
        return get_uncached_data(
            training_data_config=training_data_config,
            validation_data_config=validation_data_config,
        )
    else:
        return get_cached_data(
            training_data_config=training_data_config,
            validation_data_config=validation_data_config,
            local_download_path=local_download_path,
        )


def get_uncached_data(
    training_data_config: str, validation_data_config: Optional[str],
) -> Tuple[loaders.typing.Batches, loaders.typing.Batches]:
    with open(training_data_config, "r") as f:
        config = yaml.safe_load(f)
    loader = loaders.BatchesLoader.from_dict(config)
    logger.info("configuration loaded, creating batches object")
    train_batches = loader.load_batches()
    if validation_data_config is not None:
        with open(training_data_config, "r") as f:
            config = yaml.safe_load(f)
        loader = loaders.BatchesLoader.from_dict(config)
        logger.info("configuration loaded, creating batches object")
        val_batches = loader.load_batches()
    else:
        val_batches = []
    return train_batches, val_batches


def get_cached_data(
    training_data_config: str,
    validation_data_config: Optional[str],
    local_download_path: str,
) -> Tuple[loaders.typing.Batches, loaders.typing.Batches]:
    train_data_path = os.path.join(local_download_path, "train_data")
    logger.info("saving training data to %s", train_data_path)
    os.makedirs(train_data_path, exist_ok=True)
    save_main(data_config=training_data_config, output_path=train_data_path)
    train_batches = loaders.batches_from_netcdf(path=train_data_path)
    if validation_data_config is not None:
        validation_data_path = os.path.join(local_download_path, "validation_data")
        logger.info("saving validation data to %s", validation_data_path)
        os.makedirs(validation_data_path, exist_ok=True)
        save_main(data_config=validation_data_config, output_path=validation_data_path)
        val_batches = loaders.batches_from_netcdf(path=train_data_path)
    else:
        val_batches = []
    return train_batches, val_batches


def main(args, unknown_args=None):
    with open(args.training_config, "r") as f:
        config_dict = yaml.safe_load(f)
        if unknown_args is not None:
            config_dict = get_arg_updated_config_dict(
                args=unknown_args, config_dict=config_dict
            )
        training_config = fv3fit.TrainingConfig.from_dict(config_dict)

    with open(args.training_data_config, "r") as f:
        training_data_config = loaders.BatchesLoader.from_dict(yaml.safe_load(f))

    fv3fit.set_random_seed(training_config.random_seed)

    dump_dataclass(training_config, os.path.join(args.output_path, "train.yaml"))
    dump_dataclass(
        training_data_config, os.path.join(args.output_path, "training_data.yaml")
    )

    train_batches, val_batches = get_data(
        args.training_data_config, args.validation_data_config, args.local_download_path
    )

    train = fv3fit.get_training_function(training_config.model_type)

    logger.info("calling train function")
    model = train(
        hyperparameters=training_config.hyperparameters,
        train_batches=train_batches,
        validation_batches=val_batches,
    )
    if len(training_config.derived_output_variables) > 0:
        model = fv3fit.DerivedModel(model, training_config.derived_output_variables)
    fv3fit.dump(model, args.output_path)


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    with tempfile.NamedTemporaryFile() as temp_log:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(temp_log.name), logging.StreamHandler()],
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        main(args, unknown_args)
        copy(temp_log.name, os.path.join(args.output_path, "training.log"))
