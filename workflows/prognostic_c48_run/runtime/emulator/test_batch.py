import tensorflow as tf
from runtime.emulator.batch import batch_to_specific_humidity_basis


def test_batch_to_specific_humidity_basis(state):
    state = {key: tf.convert_to_tensor(state[key]) for key in state}
    x = batch_to_specific_humidity_basis(state, extra_inputs=["cos_zenith_angle"])
    assert len(x.scalars) == 1
