import requests
import tensorflow as tf
import xarray as xr
import pytest


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    url = "https://github.com/VulcanClimateModeling/vcm-ml-example-data/blob/b100177accfcdebff2546a396d2811e32c01c429/fv3net/prognostic_run/inputs_4x4.nc?raw=true"  # noqa
    r = requests.get(url)
    lpath = tmp_path_factory.getbasetemp() / "input_data.nc"
    lpath.write_bytes(r.content)
    return xr.open_dataset(str(lpath)).copy(deep=True)


def create_model():
    in_ = tf.keras.layers.Input(shape=(63,), name="air_temperature_input")
    out_ = tf.keras.layers.Lambda(lambda x: x + 1, name="air_temperature_dummy")(in_)
    model = tf.keras.Model(inputs=in_, outputs=out_)

    return model


@pytest.fixture(scope="session")
def saved_model_path(tmp_path_factory):
    model = create_model()
    path = tmp_path_factory.mktemp("tf_models") / "dummy_model.tf"
    model.save(path.as_posix(), save_format="tf")

    return str(path)
