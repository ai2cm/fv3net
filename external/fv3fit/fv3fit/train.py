import argparse
import inspect
import loaders
import logging
import os
import yaml

from ._shared import (
    parse_data_path,
    load_data_sequence,
    ModelTrainingConfig,
    io,
    Estimator,
)
from .keras._training import get_regularizer, get_optimizer, set_random_seed
from .keras._validation_data import validation_dataset
import fv3fit.keras
import fv3fit.sklearn


KERAS_CHECKPOINT_PATH = "model_checkpoints"
KERAS_MODEL_TYPES = [
    m[0] for m in inspect.getmembers(fv3fit.keras._models, inspect.isclass)
]
SKLEARN_MODEL_TYPES = ["sklearn", "rf", "random_forest", "sklearn_random_forest"]
ROUTINE_LOOKUP = {
    **{model: "keras" for model in KERAS_MODEL_TYPES},
    **{model: "sklearn" for model in SKLEARN_MODEL_TYPES},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_path", nargs="*", type=str, help="Location of training data"
    )
    parser.add_argument(
        "config_file", type=str, help="Local path for training configuration yaml file",
    )
    parser.add_argument(
        "output_data_path", type=str, help="Location to save config and trained model."
    )
    parser.add_argument(
        "--timesteps-file",
        type=str,
        default=None,
        help="json file containing a list of timesteps in YYYYMMDD.HHMMSS format",
    )
    parser.add_argument(
        "--validation-timesteps-file",
        type=str,
        default=None,
        help="Json file containing a list of validation timesteps in "
        "YYYYMMDD.HHMMSS format. Only relevant for keras training",
    )
    parser.add_argument(
        "--local-download-path",
        type=str,
        help="Optional path for downloading data before training. If not provided, "
        "will read from remote every epoch. Local download greatly speeds training.",
    )
    return parser.parse_args()


def _get_model(config: ModelTrainingConfig) -> Estimator:
    routine = ROUTINE_LOOKUP[config.model_type]
    if routine == "sklearn":
        return fv3fit.sklearn.get_model(
            model_type=config.model_type,
            input_variables=config.input_variables,
            output_variables=config.output_variables,
            scaler_type=config.scaler_type,
            scaler_kwargs=config.scaler_kwargs,
            **config.hyperparameters,
        )
    elif routine == "keras":
        fit_kwargs = _keras_fit_kwargs(config)
        checkpoint_path = (
            os.path.join(args.output_data_path, KERAS_CHECKPOINT_PATH)
            if config.save_model_checkpoints
            else None
        )
        return fv3fit.keras.get_model(
            model_type=config.model_type,
            sample_dim_name=loaders.SAMPLE_DIM_NAME,
            input_variables=config.input_variables,
            output_variables=config.output_variables,
            optimizer=get_optimizer(config.hyperparameters),
            kernel_regularizer=get_regularizer(config.hyperparameters),
            checkpoint_path=checkpoint_path,
            fit_kwargs=fit_kwargs,
            **config.hyperparameters,
        )
    else:
        raise NotImplementedError(f"Model type {config.model_type} is not implemented.")


def _keras_fit_kwargs(config: ModelTrainingConfig) -> dict:
    # extra args specific to keras training
    fit_kwargs = config.hyperparameters.pop("fit_kwargs", {})
    fit_kwargs["validation_dataset"] = validation_dataset(data_path, train_config)
    return fit_kwargs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    data_path = parse_data_path(args.data_path)
    train_config = ModelTrainingConfig.load(args.config_file)
    train_config.data_path = args.data_path

    if args.timesteps_file:
        with open(args.timesteps_file, "r") as f:
            timesteps = yaml.safe_load(f)
        train_config.batch_kwargs["timesteps"] = timesteps
        train_config.timesteps_source = "timesteps_file"

    if args.validation_timesteps_file:
        with open(args.validation_timesteps_file, "r") as f:
            val_timesteps = yaml.safe_load(f)
        train_config.validation_timesteps = val_timesteps

    train_config.dump(args.output_data_path)
    set_random_seed(train_config.random_seed)

    batches = load_data_sequence(data_path, train_config)
    if args.local_download_path:
        batches = batches.local(args.local_download_path)  # type: ignore

    model = _get_model(train_config)
    model.fit(batches)
    train_config.model_path = args.output_data_path
    io.dump(model, args.output_data_path)
