import tensorflow as tf
from vcm.calc.thermo import _RVGAS, _GRAVITY


def saturation_pressure(air_temperature_kelvin: tf.Tensor) -> tf.Tensor:
    """The August Roche Magnus formula for saturation vapor pressure
    
    https://en.wikipedia.org/wiki/Clausius%E2%80%93Clapeyron_relation#Meteorology_and_climatology # noqa
    
    """
    celsius = air_temperature_kelvin - 273.15
    return 610.94 * tf.exp(17.625 * celsius / (celsius + 243.04))


def relative_humidity(
    air_temperature_kelvin: tf.Tensor, specific_humidity: tf.Tensor, delp: tf.Tensor
):
    rho = delp / _GRAVITY
    partial_pressure = _RVGAS * specific_humidity * rho * air_temperature_kelvin
    return partial_pressure / saturation_pressure(air_temperature_kelvin)
