import dataclasses
from typing import Optional, Sequence, Tuple

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


class ThermoBasis:
    """A thermodynamic basis with specific humidity as the prognostic variable"""

    u: tf.Tensor
    v: tf.Tensor
    T: tf.Tensor
    q: tf.Tensor
    dp: tf.Tensor
    dz: tf.Tensor
    rh: tf.Tensor
    rho: tf.Tensor
    qc: Optional[tf.Tensor] = None
    scalars: Sequence[tf.Tensor]

    def to_rh(self):
        return RelativeHumidityBasis(
            self.u,
            self.v,
            self.T,
            self.rh,
            self.rho,
            self.dz,
            scalars=self.scalars,
            qc=self.qc,
        )

    def to_q(self):
        return SpecificHumidityBasis(
            self.u,
            self.v,
            self.T,
            self.q,
            self.dp,
            self.dz,
            scalars=self.scalars,
            qc=self.qc,
        )

    def args(self) -> Tuple[tf.Tensor]:
        raise NotImplementedError()


@dataclasses.dataclass
class SpecificHumidityBasis(ThermoBasis):
    """A thermodynamic basis with specific humidity as the prognostic variable"""

    u: tf.Tensor
    v: tf.Tensor
    T: tf.Tensor
    q: tf.Tensor
    dp: tf.Tensor
    dz: tf.Tensor
    qc: Optional[tf.Tensor] = None
    scalars: Sequence[tf.Tensor] = dataclasses.field(default_factory=list)

    @property
    def rho(self):
        return density(self.dp, self.dz)

    @property
    def rh(self) -> tf.Tensor:
        return relative_humidity(self.T, self.q, self.rho)

    @property
    def args(self):
        return (self.u, self.v, self.T, self.q, self.dp, self.dz) + tuple(self.scalars)


@dataclasses.dataclass
class RelativeHumidityBasis(ThermoBasis):

    u: tf.Tensor
    v: tf.Tensor
    T: tf.Tensor
    rh: tf.Tensor
    rho: tf.Tensor
    dz: tf.Tensor
    qc: Optional[tf.Tensor] = None
    scalars: Sequence[tf.Tensor] = dataclasses.field(default_factory=list)

    @property
    def q(self):
        return specific_humidity_from_rh(self.T, self.rh, self.rho)

    @property
    def dp(self):
        return pressure_thickness(self.rho, self.dz)

    @property
    def args(self) -> Tuple[tf.Tensor]:
        return (self.u, self.v, self.T, self.rh, self.rho, self.dz) + tuple(
            self.scalars
        )
