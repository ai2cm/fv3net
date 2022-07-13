import pytest
import tensorflow as tf

from fv3fit.emulation.transforms.zhao_carr import (
    classify,
    CLASS_NAMES,
    MicrophysicsClasssesV1,
    MicrophysicsClassesV1OneHot,
)


def test_classify():

    # Se up data so each class should be true for one feature
    # thresholds set to 1e-15 at time of writing tests
    cloud_in = tf.convert_to_tensor([[0, 0, 0, 2e-15, 3e-15]])
    cloud_out = tf.convert_to_tensor([[1e-14, 1e-16, -1e-15, 1e15, 2e-15]])

    result = classify(cloud_in, cloud_out, timestep=1.0)

    class_sum = tf.reduce_sum(
        [tf.cast(classified, tf.int16) for classified in result.values()], axis=0
    )
    tf.debugging.assert_equal(class_sum, tf.ones(5, dtype=tf.int16))


def _get_cloud_state(shape=(4, 2)):
    return {
        "cloud_water_mixing_ratio_input": tf.zeros(shape),
        "cloud_water_mixing_ratio_after_gscond": tf.ones(shape),
    }


@pytest.mark.parametrize(
    "classify_class", [MicrophysicsClasssesV1, MicrophysicsClassesV1OneHot],
)
def test_classify_classes_build(classify_class):

    state = _get_cloud_state()

    factory = classify_class()
    transform = factory.build(state)
    assert transform


def test_classify_classes_v1():
    state = _get_cloud_state()

    transform = MicrophysicsClasssesV1().build(state)

    required = transform.backward_names(CLASS_NAMES)
    assert transform.cloud_in in required
    assert transform.cloud_out in required

    new_state = transform.forward(state)
    assert set(new_state) & CLASS_NAMES == CLASS_NAMES


def test_classify_classes_v1OneHot():
    state = _get_cloud_state()

    factory = MicrophysicsClassesV1OneHot()
    transform = factory.build(state)

    required = factory.backward_names({transform.to})
    assert factory.cloud_in in required
    assert factory.cloud_out in required

    new_state = transform.forward(state)
    assert factory.to in new_state
    assert new_state[factory.to].shape[-1] == len(CLASS_NAMES)
