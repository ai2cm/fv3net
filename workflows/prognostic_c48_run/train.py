#!/usr/bin/env python3
from functools import partial
import json
import os

from runtime.emulator.emulator import Config as OnlineEmulatorConfig
from runtime.emulator.batch import all_required_variables, compute_in_out, nz
from fv3fit.emulation import data
import runtime.emulator
import tensorflow as tf
import pathlib

import logging

import wandb
import argparse

logging.basicConfig(level=logging.INFO)


def main(config: OnlineEmulatorConfig):

    tf.random.set_seed(1)
    logging.info(config)

    if config.batch is None:
        raise ValueError("No training dataset detected.")

    if config.wandb_logger:
        wandb.init(
            entity="ai2cm", project=f"emulator-noah", config=args,  # type: ignore
        )

    emulator = runtime.emulator.OnlineEmulator(config)

    prep = partial(compute_in_out, timestep=args.timestep)
    variables = config.extra_input_variables + all_required_variables()

    train_dataset = data.netcdf_url_to_dataset(
        config.batch.training_path, variables, shuffle=True,
    ).map(prep)

    test_dataset = data.netcdf_url_to_dataset(
        config.batch.testing_path, variables,
    ).map(prep)

    if args.nfiles:
        train_dataset = train_dataset.take(args.nfiles)
        test_dataset = test_dataset.take(args.nfiles)

    train_dataset = train_dataset.unbatch().cache()
    test_dataset = test_dataset.unbatch().cache()

    # detect number of levels
    sample_ins, _ = next(iter(train_dataset.batch(10).take(1)))
    config.levels = nz(sample_ins)

    id_ = pathlib.Path(os.getcwd()).name

    with tf.summary.create_file_writer(f"/data/emulator/{id_}").as_default():
        emulator.batch_fit(train_dataset.shuffle(100_000), validation_data=test_dataset)

    train_scores = emulator.score(train_dataset)
    test_scores = emulator.score(test_dataset)

    if config.output_path:
        os.makedirs(config.output_path, exist_ok=True)

    with open(os.path.join(config.output_path, "scores.json"), "w") as f:
        json.dump({"train": train_scores, "test": test_scores}, f)

    emulator.dump(os.path.join(config.output_path, "model"))

    if config.wandb_logger:
        model = wandb.Artifact(f"model", type="model")
        model.add_dir(os.path.join(config.output_path, "model"))
        wandb.log_artifact(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    OnlineEmulatorConfig.register_parser(parser)
    args = parser.parse_args()
    config = OnlineEmulatorConfig.from_args(args)
    print(config)
    main(config)
