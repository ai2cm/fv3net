"""
k8s entrypoint for emulation
"""

import argparse
import logging
from fv3fit.keras._models.classifiers import DenseClassifierModel
import tensorflow as tf
from pathlib import Path

from fv3fit.keras import get_model
from fv3fit._shared import load_model_training_config
from loaders.batches import batches_from_serialized
from loaders import shuffle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_path",
        type=str,
        help="Location of serialized physics data to train on",
    )
    parser.add_argument(
        "train_config_file",
        type=str,
        help="Local path for training configuration yaml file",
    )
    parser.add_argument(
        "output_data_path", type=str, help="Location to save config and trained model"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)

    config = load_model_training_config(args.train_config_file, args.train_data_path,)

    train_range = config.batch_kwargs.get("train_range", (None,))
    seed = config.batch_kwargs.get("seed", None)
    batches = batches_from_serialized(args.train_data_path)
    train = shuffle(batches[slice(*train_range)], seed=seed)

    hyper_params = config.hyperparameters
    fit_kwargs = config.hyperparameters.pop("fit_kwargs", {})

    # Handle optimizer
    optimizer_name = hyper_params.pop("optimizer", "Adam")
    optimizer_class = getattr(tf.keras.optimizers, optimizer_name)
    optimizer_kwargs = {}
    lr = hyper_params.pop("learning_rate", None)
    if lr == "exponential":
        lr = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, 10_000, 0.96)
    optimizer_kwargs["learning_rate"] = lr
    optimizer = optimizer_class(**optimizer_kwargs)
    hyper_params["optimizer"] = optimizer

    # Handle classifiers
    classifier_paths = hyper_params.pop("classifiers", None)
    if classifier_paths is not None:
        classifiers = {
            varname: DenseClassifierModel.load(path)
            for varname, path in classifier_paths.items()
        }
        hyper_params["classifiers"] = classifiers

    # TODO is var name handling okay?
    sample = train[0]
    input_vars = [var for var in sample if "input" in var]
    output_vars = [var for var in sample if "output" in var]

    model = get_model(
        config.model_type,
        "sample",
        input_vars if config.input_variables is None else config.input_variables,
        output_vars if config.output_variables is None else config.output_variables,
        **hyper_params,
    )
    model.fit(train, **fit_kwargs)

    model_output_path = Path(args.output_data_path, "keras_model")
    model.dump(str(model_output_path.resolve()))
