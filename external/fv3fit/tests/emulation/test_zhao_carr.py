import tensorflow as tf

from fv3fit.emulation.transforms.zhao_carr import classify


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
