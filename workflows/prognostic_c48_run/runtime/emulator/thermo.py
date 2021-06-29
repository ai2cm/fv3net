import tensorflow as tf
from vcm.calc.thermo import _RVGAS, _GRAVITY


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
    def rh(self):
        raise NotImplementedError()

    @property
    def rho(self):
        raise NotImplementedError()

    @property
    def T(self):
        raise NotImplementedError()

    @property
    def dz(self):
        raise NotImplementedError()

    @property
    def dp(self):
        raise NotImplementedError()

    def to_rh(self):
        return RelativeHumidityBasis(
            (self.u, self.v, self.T, self.rh, self.rho, self.dz) + self.args[6:]
        )

    def to_q(self):
        return SpecificHumidityBasis(
            (self.u, self.v, self.T, self.q, self.dp, self.dz, *self.args[6:])
        )


class SpecificHumidityBasis(ThermoBasis):
    """A thermodynamic basis with specific humidity as the prognostic variable"""

    def __init__(self, args):
        self.args = args

    @property
    def u(self):
        return self.args[0]

    @property
    def v(self):
        return self.args[1]

    @property
    def T(self):
        return self.args[2]

    @property
    def q(self):
        return self.args[3]

    @property
    def dp(self):
        return self.args[4]

    @property
    def dz(self):
        return self.args[5]

    @property
    def rho(self):
        return density(self.dp, self.dz)

    @property
    def rh(self) -> tf.Tensor:
        return relative_humidity(self.T, self.q, self.rho)


class RelativeHumidityBasis(SpecificHumidityBasis):
    def __init__(self, args):
        self.args = args

    @property
    def rh(self):
        return self.args[3]

    @property
    def q(self):
        return specific_humidity_from_rh(self.T, self.rh, self.rho)

    @property
    def dp(self):
        return pressure_thickness(self.rho, self.dz)

    @property
    def rho(self):
        return self.args[4]
