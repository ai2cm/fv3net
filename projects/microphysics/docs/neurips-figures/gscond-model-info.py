# flake8: noqa
# %%
import tensorflow as tf
import fv3fit.train_microphysics as microphysics


def load_model_and_config(url):
    url = url.rstrip("/")
    model_url = url + "/model.tf"
    config_url = url + "/config.yaml"
    model = tf.keras.models.load_model(model_url)
    config = microphysics.TrainConfig.from_yaml_path(config_url)
    return model, config


def report_info(model, config):
    print(model.summary())

    print(config.model.input_variables)
    print(len(config.model.input_variables))
    print(config.model.normalize_default)
    print(config.model.architecture)
    print(config.model.normalize_map)
    print(config.loss.loss_variables)
    print(config.loss.weights)
    print("Training details.")
    print(config.loss.optimizer)
    print(config.batch_size)
    print(config.batch_size * 79)
    print(config.epochs)


print("GSCOND MODEL")
url = "gs://vcm-ml-experiments/microphysics-emulation/2022-05-12/gscond-only-tscale-dense-local-41b1c1-v1"
model, config = load_model_and_config(url)
report_info(model, config)

print("PRECPD MODEL")
url = "gs://vcm-ml-experiments/microphysics-emulation/2022-07-19/precpd-diff-only-rnn-v1-shared-weights-v1"
model, config = load_model_and_config(url)
report_info(model, config)
# %%
