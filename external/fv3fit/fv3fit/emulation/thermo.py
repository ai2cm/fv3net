import tensorflow as tf
from vcm.calc.thermo import _GRAVITY, _RVGAS


def saturation_pressure(air_temperature_kelvin: tf.Tensor) -> tf.Tensor:
    """The August Roche Magnus formula for saturation vapor pressure
    
    https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology # noqa
    
    """
    celsius = air_temperature_kelvin - 273.15
    return 610.94 * tf.exp(17.625 * celsius / (celsius + 243.04))


def relative_humidity(
    air_temperature_kelvin: tf.Tensor, specific_humidity: tf.Tensor, rho: tf.Tensor
):
    partial_pressure = _RVGAS * specific_humidity * rho * air_temperature_kelvin
    return partial_pressure / saturation_pressure(air_temperature_kelvin)


def specific_humidity_from_rh(
    air_temperature_kelvin, relative_humidity, rho: tf.Tensor
):
    es = saturation_pressure(air_temperature_kelvin)
    partial_pressure = relative_humidity * es

    return partial_pressure / _RVGAS / rho / air_temperature_kelvin


def density(delp, delz):
    return tf.abs(delp / delz / _GRAVITY)


def pressure_thickness(rho, delz):
    return tf.abs(rho * delz * _GRAVITY)
