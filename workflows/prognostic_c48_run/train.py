import data
import runtime.emulator
import tensorflow as tf
import uuid

tf.random.set_seed(1)

config = runtime.emulator.OnlineEmulatorConfig(
    epochs=100, learning_rate=0.01, q_weight=1e10, t_weight=1e3
)
emulator = runtime.emulator.OnlineEmulator(config)

timestep = 900

train_dataset = (
    data.netcdf_url_to_dataset(
        "gs://vcm-ml-scratch/andrep/all-physics-emu/training-subsets/simple-phys-hybridedmf-10day",  # noqa
        timestep,
        emulator.input_variables,
    )
    .take(20)
    .unbatch()
    .shuffle(100_000)
    .cache("train")
)
test_dataset = (
    data.netcdf_url_to_dataset(
        "gs://vcm-ml-scratch/andrep/all-physics-emu/validation-subsets/simple-phys-hybridedmf-10day",  # noqa
        timestep,
        emulator.input_variables,
    )
    .take(10)
    .unbatch()
    .shuffle(100_000)
    .cache("test")
)

with tf.summary.create_file_writer(f"tensorboard/{uuid.uuid4().hex}").as_default():
    emulator.batch_fit(train_dataset, validation_data=test_dataset)
