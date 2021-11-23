import tensorflow as tf
from vcm.calc.thermo import _GRAVITY, _RVGAS


def saturation_pressure(air_temperature_kelvin: tf.Tensor) -> tf.Tensor:
    """The August Roche Magnus formula for saturation vapor pressure
    
    https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology # noqa
    
    """
    celsius = air_temperature_kelvin - 273.15
    return 610.94 * tf.exp(17.625 * celsius / (celsius + 243.04))


def relative_humidity(
    air_temperature_kelvin: tf.Tensor, specific_humidity: tf.Tensor, density: tf.Tensor
):
    partial_pressure = _RVGAS * specific_humidity * density * air_temperature_kelvin
    return partial_pressure / saturation_pressure(air_temperature_kelvin)


def specific_humidity_from_rh(
    air_temperature_kelvin, relative_humidity, density: tf.Tensor
):
    es = saturation_pressure(air_temperature_kelvin)
    partial_pressure = relative_humidity * es

    return partial_pressure / _RVGAS / density / air_temperature_kelvin


def density(delp, delz):
    return tf.abs(delp / delz / _GRAVITY)


def pressure_thickness(density, delz):
    return tf.abs(density * delz * _GRAVITY)


def mass_integrate(x, delp, axis=0):
    return tf.reduce_sum(x * delp, axis) / _GRAVITY


def layer_mass(delp: tf.Tensor) -> tf.Tensor:
    """Layer mass in kg/m^2 from ``delp`` in Pa"""
    return delp / _GRAVITY


def conservative_precipitation_zhao_carr(
    specific_humidity_before: tf.Tensor,
    specific_humidity_after: tf.Tensor,
    cloud_before: tf.Tensor,
    cloud_after: tf.Tensor,
    mass: tf.Tensor,
    vertical_axis: int = -1,
) -> tf.Tensor:
    water_before = specific_humidity_before + cloud_before
    water_after = specific_humidity_after + cloud_after
    column_water_before = tf.reduce_sum(water_before * mass, axis=vertical_axis)
    column_water_after = tf.reduce_sum(water_after * mass, axis=vertical_axis)
    return column_water_before - column_water_after
