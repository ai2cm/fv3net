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


def specific_humidity_from_rh(air_temperature_kelvin, relative_humidity, delp):
    rho = delp / _GRAVITY

    es = saturation_pressure(air_temperature_kelvin)
    partial_pressure = relative_humidity * es

    return partial_pressure / _RVGAS / rho / air_temperature_kelvin


class ThermoBasis:
    """A thermodynamic basis with specific humidity as the prognostic variable"""

    @property
    def u(self):
        raise NotImplementedError()

    @property
    def v(self):
        raise NotImplementedError()

    @property
    def q(self):
        raise NotImplementedError()

    @property
    def dp(self):
        raise NotImplementedError()

    @property
    def T(self):
        raise NotImplementedError()


class SpecificHumidityBasis(ThermoBasis):
    """A thermodynamic basis with specific humidity as the prognostic variable"""

    def __init__(self, args):
        self.args = args

    @property
    def u(self):
        return self.args[0]

    @property
    def v(self):
        return self.args[0]

    @property
    def q(self):
        return self.args[3]

    @property
    def dp(self):
        return self.args[4]

    @property
    def T(self):
        return self.args[2]

    @property
    def rh(self) -> tf.Tensor:
        return relative_humidity(self.T, self.q, self.dp)

    def to_rh(self):
        return RelativeHumidityBasis(
            (self.u, self.v, self.rh, self.T, self.dp) + self.args[5:]
        )


class RelativeHumidityBasis(SpecificHumidityBasis):
    def __init__(self, args):
        self.args = args

    @property
    def rh(self):
        return self.args[3]

    @property
    def q(self):
        return specific_humidity_from_rh(self.T, self.rh, self.dp)

    def to_q(self):
        return SpecificHumidityBasis(
            (self.u, self.v, self.q, self.T, self.dp, *self.args[5:])
        )
