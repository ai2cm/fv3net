import argparse
import logging
import tempfile
import os
from typing import Optional, Sequence, Tuple
import tensorflow as tf
from fv3fit._shared.config import CacheConfig
from fv3fit._shared.training_config import (
    get_arg_updated_config_dict,
    to_flat_dict,
    to_nested_dict,
)
from fv3fit._shared import put_dir

import yaml
import fsspec

import fv3fit.keras
import fv3fit.sklearn
import fv3fit
from .data import tfdataset_loader_from_dict, TFDatasetLoader
from fv3fit.dataclasses import asdict_with_enum
import wandb
import sys
import shutil

from fv3net.artifacts.metadata import StepMetadata

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
        "--no-wandb",
        help=(
            "Disable logging of run to wandb. Uses environment variables WANDB_ENTITY, "
            "WANDB_PROJECT, WANDB_JOB_TYPE as wandb.init options."
        ),
        action="store_true",
    )
    return parser


def dump_dataclass(obj, yaml_filename):
    with fsspec.open(yaml_filename, "w") as f:
        yaml.safe_dump(asdict_with_enum(obj), f)


def maybe_join_path(base: Optional[str], append: str) -> Optional[str]:
    if base is not None:
        return os.path.join(base, append)
    else:
        return None


def load_data(
    variables: Sequence[str],
    training_data_config: TFDatasetLoader,
    validation_data_config: Optional[TFDatasetLoader],
    cache_config: CacheConfig,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
    train_tfdataset = training_data_config.open_tfdataset(
        local_download_path=maybe_join_path(
            cache_config.local_download_path, "train_data"
        ),
        variable_names=variables,
    )
    if cache_config.in_memory:
        train_tfdataset = train_tfdataset.cache()
    if validation_data_config is not None:
        validation_tfdataset = validation_data_config.open_tfdataset(
            local_download_path=maybe_join_path(
                cache_config.local_download_path, "validation_data"
            ),
            variable_names=variables,
        )
        if cache_config.in_memory:
            validation_tfdataset = validation_tfdataset.cache()
    else:
        validation_tfdataset = None
    return train_tfdataset, validation_tfdataset


def main(args, unknown_args=None):
    with open(args.training_config, "r") as f:
        config_dict = yaml.safe_load(f)
        if unknown_args is not None:
            # converting to TrainingConfig and then back to dict allows command line to
            # update fields that are not present in original configuration file
            config_dict = asdict_with_enum(fv3fit.TrainingConfig.from_dict(config_dict))
            config_dict = get_arg_updated_config_dict(
                args=unknown_args, config_dict=config_dict
            )
        if args.no_wandb is False:
            wandb.init(
                # workaround for
                # https://docs.wandb.ai/guides/track/launch#init-start-error
                settings=wandb.Settings(start_method="fork"),
                # hyperparameters are repeated as flattened top level keys so they can
                # be referenced in the sweep configuration parameters
                # https://github.com/wandb/client/issues/982
                config=to_flat_dict(config_dict["hyperparameters"]),
            )
            # hyperparameters should be accessed throughthe wandb config so that
            # sweeps use the wandb-provided hyperparameter values
            config_dict["hyperparameters"] = to_nested_dict(wandb.config)
            logger.info(
                f"hyperparameters from wandb config: {config_dict['hyperparameters']}"
            )
            wandb.config["training_config"] = config_dict
            wandb.config["env"] = {"COMMIT_SHA": os.getenv("COMMIT_SHA", "")}

        training_config = fv3fit.TrainingConfig.from_dict(config_dict)

    with open(args.training_data_config, "r") as f:
        config_dict = yaml.safe_load(f)
        training_data_config = tfdataset_loader_from_dict(config_dict)
        if args.no_wandb is False:
            wandb.config["training_data_config"] = config_dict
    if args.validation_data_config is not None:
        with open(args.validation_data_config, "r") as f:
            config_dict = yaml.safe_load(f)
            validation_data_config = tfdataset_loader_from_dict(config_dict)
            if args.no_wandb is False:
                wandb.config["validation_data_config"] = config_dict
    else:
        validation_data_config = None

    fv3fit.set_random_seed(training_config.random_seed)

    dump_dataclass(training_config, os.path.join(args.output_path, "train.yaml"))
    dump_dataclass(
        training_data_config, os.path.join(args.output_path, "training_data.yaml")
    )

    train_tfdataset, validation_tfdataset = load_data(
        training_config.variables,
        training_data_config,
        validation_data_config,
        training_config.cache,
    )

    train = fv3fit.get_training_function(training_config.model_type)

    logger.info("calling train function")
    model = train(
        hyperparameters=training_config.hyperparameters,
        train_batches=train_tfdataset,
        validation_batches=validation_tfdataset,
    )
    if len(training_config.derived_output_variables) > 0:
        model = fv3fit.DerivedModel(model, training_config.derived_output_variables)
    if len(training_config.output_transforms) > 0:
        model = fv3fit.TransformedPredictor(model, training_config.output_transforms)
    fv3fit.dump(model, args.output_path)
    StepMetadata(
        job_type="training", url=args.output_path, args=sys.argv[1:],
    ).print_json()


def disable_tensorflow_gpu_preallocation():
    """
    Enables "memory growth" option on all gpus for tensorflow.

    Without this, tensorflow will eagerly allocate all gpu memory,
    leaving none for pytorch.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        logging.info("%d physical gpus, %d logical gpus", len(gpus), len(logical_gpus))


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    disable_tensorflow_gpu_preallocation()
    dir_ = tempfile.TemporaryDirectory()
    os.makedirs(os.path.join(dir_.name, "artifacts"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s::L%(lineno)d : %(message)s",
        handlers=[
            logging.FileHandler(f"{dir_.name}/artifacts/training.log"),
            logging.StreamHandler(),
        ],
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args, unknown_args)
    with put_dir(args.output_path) as path:
        shutil.move(
            os.path.join(dir_.name, "artifacts"), os.path.join(path, "artifacts")
        )
