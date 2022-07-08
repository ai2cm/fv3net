import dataclasses
from typing import Set, Optional

import tensorflow as tf
from fv3fit.emulation.types import TensorDict
from fv3fit.emulation.transforms import TensorTransform
import vcm


@dataclasses.dataclass
class TendencyToFlux(TensorTransform):
    """
    From an array of cell-centered tendencies, TOA net flux and upward surface flux,
    compute vertical fluxes at cell interfaces and downward surface flux.
    """

    tendency: str
    interface_flux: str
    down_sfc_flux: str
    up_sfc_flux: str
    delp: str
    net_toa_flux: Optional[str] = None  # if not provided, assume TOA flux is zero
    gravity: float = 9.8065  # TODO: define vcm mass integral function for tensors

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        if self.interface_flux in requested_names:
            requested_names -= {self.interface_flux}
            requested_names |= {self.tendency, self.up_sfc_flux, self.delp}
            if self.net_toa_flux:
                requested_names |= {self.net_toa_flux}

        if self.down_sfc_flux in requested_names:
            requested_names -= {self.down_sfc_flux}
            requested_names |= {self.tendency, self.up_sfc_flux, self.delp}
            if self.net_toa_flux:
                requested_names |= {self.net_toa_flux}

        return requested_names

    def backward_input_names(self) -> Set[str]:
        names = {self.interface_flux, self.down_sfc_flux, self.up_sfc_flux, self.delp}
        if self.net_toa_flux is not None:
            names |= {self.net_toa_flux}
        return names

    def backward_output_names(self) -> Set[str]:
        return {self.tendency}

    def forward(self, y: TensorDict):
        y = {**y}
        flux = tf.constant(-1 / self.gravity, dtype=tf.float32) * tf.math.cumsum(
            y[self.tendency] * y[self.delp], axis=-1, exclusive=True
        )
        net_sfc_flux = tf.constant(
            -1 / self.gravity, dtype=tf.float32
        ) * tf.math.reduce_sum(y[self.tendency] * y[self.delp], axis=-1, keepdims=True)

        if self.net_toa_flux is not None:
            flux += y[self.net_toa_flux]
            net_sfc_flux += y[self.net_toa_flux]

        y[self.interface_flux] = flux
        y[self.down_sfc_flux] = net_sfc_flux + y[self.up_sfc_flux]
        return y

    def backward(self, x: TensorDict):
        x = {**x}

        net_sfc_flux = x[self.down_sfc_flux] - x[self.up_sfc_flux]
        all_fluxes = tf.concat([x[self.interface_flux], net_sfc_flux], axis=-1)
        x[self.tendency] = (
            tf.constant(-self.gravity, dtype=tf.float32)
            * (all_fluxes[..., 1:] - all_fluxes[..., :-1])
            / x[self.delp]
        )
        return x


@dataclasses.dataclass
class MoistStaticEnergyTransform(TensorTransform):
    """
    From heating (in K/s) and moistening (in kg/kg/s) rates, compute moist static energy
    tendency in W/kg. The backwards transformation computes heating from moistening and
    MSE tendnecy.
    """

    heating: str
    moistening: str
    mse_tendency: str

    def build(self, sample: TensorDict) -> TensorTransform:
        return self

    def backward_names(self, requested_names: Set[str]) -> Set[str]:

        if self.mse_tendency in requested_names:
            requested_names -= {self.mse_tendency}
            requested_names |= {self.heating, self.moistening}

        return requested_names

    def forward(self, y: TensorDict):
        y = {**y}
        y[self.mse_tendency] = vcm.moist_static_energy_tendency(
            y[self.heating], y[self.moistening]
        )
        return y

    def backward(self, x: TensorDict):
        x = {**x}
        x[self.heating] = vcm.temperature_tendency(
            x[self.mse_tendency], x[self.moistening]
        )
        return x
