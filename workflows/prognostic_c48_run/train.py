import json
import os
import data
import runtime.emulator
import tensorflow as tf
import uuid
import dacite
import yaml
import sys

tf.random.set_seed(1)

with open(sys.argv[1]) as f:
    dict_ = yaml.safe_load(f)

config = dacite.from_dict(runtime.emulator.OnlineEmulatorConfig, dict_)
emulator = runtime.emulator.OnlineEmulator(config)

timestep = 900

train_dataset = (
    data.netcdf_url_to_dataset(
        config.batch.training_path, timestep, emulator.input_variables,
    )
    .take(20)
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


with tf.summary.create_file_writer(f"tensorboard/{uuid.uuid4().hex}").as_default():
    emulator.batch_fit(train_dataset, validation_data=test_dataset)

train_scores = emulator.score(train_dataset)
test_scores = emulator.score(test_dataset)

with open(os.path.join(config.output_path, "scores.json"), "w") as f:
    json.dump({"train": train_scores, "test": test_scores}, f)
