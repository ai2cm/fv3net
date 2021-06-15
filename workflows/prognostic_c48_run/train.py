import json
import os
import data
import runtime.emulator
import tensorflow as tf
import pathlib

import logging

logging.basicConfig(level=logging.INFO)

tf.random.set_seed(1)

# with open(sys.argv[1]) as f:
#     dict_ = yaml.safe_load(f)

batch_size = 32
epochs = 60
lr = 0.01
timestep = 900
train_path = "data/training"
test_path = "data/validation"
problem = "single-level"
scale = 1e-9

# config = runtime.emulator.OnlineEmulatorConfig.from_dict(dict_)
config = runtime.emulator.OnlineEmulatorConfig()
config.batch_size = batch_size
config.epochs = epochs
config.learning_rate = lr
config.batch = runtime.emulator.BatchDataConfig(train_path, test_path)
if problem == "single-level":
    config.target = runtime.emulator.ScalarLoss(3, 50, scale=scale)

logging.info(config)
emulator = runtime.emulator.OnlineEmulator(config)


train_dataset = (
    data.netcdf_url_to_dataset(
        config.batch.training_path, timestep, emulator.input_variables,
    )
    .unbatch()
    .shuffle(100_000)
    .cache()
)
test_dataset = (
    data.netcdf_url_to_dataset(
        config.batch.testing_path, timestep, emulator.input_variables,
    )
    .unbatch()
    .cache()
)

id_ = pathlib.Path(os.getcwd()).name

with tf.summary.create_file_writer(f"/data/emulator/{id_}").as_default():
    emulator.batch_fit(train_dataset, validation_data=test_dataset)

train_scores = emulator.score(train_dataset)
test_scores = emulator.score(test_dataset)

if config.output_path:
    os.makedirs(config.output_path, exist_ok=True)

with open(os.path.join(config.output_path, "scores.json"), "w") as f:
    json.dump({"train": train_scores, "test": test_scores}, f)

emulator.dump(os.path.join(config.output_path, "model"))
