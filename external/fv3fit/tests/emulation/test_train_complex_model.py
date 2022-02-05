from fv3fit.emulation.models.complex_model import compute_gscond, compute_precpd
import tensorflow as tf


def test_compute_gscond():
    n = 4

    def make_input(name=None):
        return tf.keras.Input(n, name=name)

    def make_sample():
        return tf.random.uniform(shape=(10, n))

    specific_humidity_in = make_input()
    temperature_in = make_input()
    cloud_in = make_input()

    specific_humidity_sample = make_sample()
    temperature_sample = make_sample()
    latent_heat_sample = make_sample()
    cloud_sample = make_sample()
    cloud_to_vapor_sample = make_sample()

    qv, t, qc, lv = compute_gscond(
        specific_humidity_in,
        temperature_in,
        cloud_in,
        [],
        specific_humidity_sample,
        temperature_sample,
        cloud_sample,
        cloud_to_vapor_sample,
        latent_heat_sample,
        [],
    )


def test_compute_precpd():
    n = 4

    def make_input(name=None):
        return tf.keras.Input(n, name=name)

    def make_sample():
        return tf.random.uniform(shape=(10, n))

    specific_humidity_in = make_input()
    temperature_in = make_input()
    cloud_in = make_input()

    specific_humidity_sample = make_sample()
    temperature_sample = make_sample()
    latent_heat_sample = make_sample()
    cloud_sample = make_sample()
    vapor_change_sample = make_sample()
    cloud_change_sample = make_sample()

    qv, t, qc = compute_precpd(
        specific_humidity_in,
        temperature_in,
        cloud_in,
        [],
        specific_humidity_sample,
        temperature_sample,
        cloud_sample,
        vapor_change_sample,
        cloud_change_sample,
        latent_heat_sample,
        [],
    )
