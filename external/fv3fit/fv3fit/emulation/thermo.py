import tensorflow as tf
import vcm


def mass_integrate(x, delp, axis=0):
    return tf.reduce_sum(x * vcm.layer_mass(delp), axis)


def liquid_water_equivalent(f: tf.Tensor) -> tf.Tensor:
    """f has units proportional to kg/m^2

    Returns:
        f with units proportional to m

    """
    density_liquid_water = 1000.0
    return f / density_liquid_water


def conservative_precipitation_zhao_carr(
    specific_humidity_before: tf.Tensor,
    specific_humidity_after: tf.Tensor,
    cloud_before: tf.Tensor,
    cloud_after: tf.Tensor,
    mass: tf.Tensor,
    vertical_axis: int = -1,
) -> tf.Tensor:
    """Compute accumulated precipitation in meters"""
    water_before = specific_humidity_before + cloud_before
    water_after = specific_humidity_after + cloud_after
    column_water_before = tf.reduce_sum(water_before * mass, axis=vertical_axis)
    column_water_after = tf.reduce_sum(water_after * mass, axis=vertical_axis)
    return liquid_water_equivalent(column_water_before - column_water_after)
