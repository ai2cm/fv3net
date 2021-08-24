from typing import Optional, Sequence
import dataclasses
import tensorflow as tf
from fv3fit.emulation.thermo import (
    density,
    relative_humidity,
    specific_humidity_from_rh,
    pressure_thickness,
)


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

    def to_rh(self) -> "RelativeHumidityBasis":
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

    def to_q(self) -> "SpecificHumidityBasis":
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

    @property
    def args(self) -> Sequence[tf.Tensor]:
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
    def rho(self) -> tf.Tensor:
        return density(self.dp, self.dz)

    @property
    def rh(self) -> tf.Tensor:
        return relative_humidity(self.T, self.q, self.rho)

    @property
    def args(self) -> Sequence[tf.Tensor]:
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
    def q(self) -> tf.Tensor:
        return specific_humidity_from_rh(self.T, self.rh, self.rho)

    @property
    def dp(self) -> tf.Tensor:
        return pressure_thickness(self.rho, self.dz)

    @property
    def args(self) -> Sequence[tf.Tensor]:
        return (self.u, self.v, self.T, self.rh, self.rho, self.dz) + tuple(
            self.scalars
        )
