import tensorflow as tf
from fv3fit.train_microphysics import HighAntarctic
from fv3fit.keras.adapters import ensure_dict_output
import wandb
import fv3net.artifacts.resolve_url

job = wandb.init(job_type="combine", project="microphysics-emulation", entity="ai2cm")


inputs = {
    "specific_humidity_input": tf.TensorSpec(shape=(79,), dtype=tf.float32, name=None),
    "specific_humidity_after_last_gscond": tf.TensorSpec(
        shape=(79,), dtype=tf.float32, name=None
    ),
    "air_temperature_input": tf.TensorSpec(shape=(79,), dtype=tf.float32, name=None),
    "air_pressure": tf.TensorSpec(shape=(79,), dtype=tf.float32, name=None),
    "air_temperature_after_last_gscond": tf.TensorSpec(
        shape=(79,), dtype=tf.float32, name=None
    ),
    "pressure_thickness_of_atmospheric_layer": tf.TensorSpec(
        shape=(79,), dtype=tf.float32, name=None
    ),
    "cloud_water_mixing_ratio_input": tf.TensorSpec(
        shape=(79,), dtype=tf.float32, name=None
    ),
    "surface_air_pressure": tf.TensorSpec(shape=(1,), dtype=tf.float32, name=None),
    "latitude": tf.TensorSpec(shape=(1,), dtype=tf.float32, name=None),
    "surface_air_pressure_after_last_gscond": tf.TensorSpec(
        shape=(1,), dtype=tf.float32, name=None
    ),
}

outputs = {
    "specific_humidity_after_gscond": tf.TensorSpec(
        shape=(79,), dtype=tf.float32, name=None
    )
}


def spec_to_inputs(spec):
    return {
        name: tf.keras.Input(input.shape[0], dtype=input.dtype, name=name)
        for name, input in inputs.items()
    }


NAME = "combined-gscond-dense-local-75s"
MODEL_REST = "gs://vcm-ml-experiments/microphysics-emulation/2022-08-17/combined-gscond-dense-local-v2/model.tf"  # noqa
MODEL_ANTARCTIC = "ai2cm/microphysics-emulation/microphysics-emulator-dense-local:v28"
max_lat = -75


def open_model(path):
    if path.startswith("ai2cm"):
        artifact = job.use_artifact(path)
        dir_ = artifact.download()
        return tf.keras.models.load_model(dir_)
    else:
        return tf.keras.models.load_model(path)


model_antarctic = open_model(MODEL_ANTARCTIC)
model_rest = open_model(MODEL_REST)

model_antarctic = ensure_dict_output(model_antarctic)
model_rest = ensure_dict_output(model_rest)

# need to set names to avoid internal layer conflicts
model_antarctic._name = "antarctic-75s"
model_rest._name = "rest"

keras_inputs = spec_to_inputs(inputs)
out_rest = model_rest(keras_inputs)
out_antartic = model_antarctic(keras_inputs)

latitude = keras_inputs["latitude"]
mask = HighAntarctic(max_lat=max_lat).mask(keras_inputs)
out_merged = [
    tf.keras.layers.Lambda(lambda x: x, name=key)(
        tf.where(mask, out_antartic[key], out_rest[key])
    )
    for key in out_antartic
]
combined = tf.keras.Model(inputs=keras_inputs, outputs=out_merged)
out_url = fv3net.artifacts.resolve_url.resolve_url(
    "vcm-ml-experiments", job.project, NAME
)
combined.save(f"{out_url}/model.tf")
artifact = wandb.Artifact(NAME, type="gscond-model")
artifact.add_reference(f"{out_url}/model.tf", checksum=False)
job.log_artifact(artifact)
