import dataclasses
from typing import Mapping, Iterable, Hashable

import numpy as np
import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import non_negative_sphum
from runtime.types import State
from runtime.names import SPHUM

__all__ = ["Config", "Adapter"]

# these were computed in a separate notebook
OFFLINE_BIASES = {
    "Q1": xr.DataArray(
        np.array(
            [
                -5.95483844e-07,
                3.34154064e-06,
                -8.69573496e-08,
                -2.34940215e-06,
                -2.23262500e-06,
                -2.53265038e-06,
                -2.41729968e-06,
                -2.70158420e-06,
                -2.76445362e-06,
                -2.17168922e-06,
                -1.32704781e-06,
                -1.06097486e-06,
                -1.20098640e-06,
                -1.12790619e-06,
                -1.35132457e-06,
                -1.37707434e-06,
                -2.95575989e-07,
                8.20116608e-07,
                1.68773040e-06,
                1.77378498e-06,
                1.83620161e-06,
                1.99403880e-06,
                2.03939758e-06,
                2.05850884e-06,
                2.10648974e-06,
                2.25930862e-06,
                2.53792922e-06,
                2.82958718e-06,
                2.99945241e-06,
                3.08254360e-06,
                3.04533017e-06,
                3.19073828e-06,
                3.20659510e-06,
                3.12808986e-06,
                3.02528151e-06,
                2.94470821e-06,
                2.92360001e-06,
                2.92973013e-06,
                2.99458093e-06,
                3.02739120e-06,
                3.07742052e-06,
                3.08808131e-06,
                3.01049457e-06,
                2.77295499e-06,
                2.34175437e-06,
                1.73208055e-06,
                1.06255487e-06,
                3.12410861e-07,
                -1.88520617e-07,
                -5.29314375e-07,
                -7.35637558e-07,
                -7.93832829e-07,
                -8.64981531e-07,
                -6.03955424e-07,
                -5.36059473e-07,
                -5.36711686e-07,
                -5.50934614e-07,
                -5.49552236e-07,
                -4.95407713e-07,
                -3.48676802e-07,
                -8.94323467e-08,
                1.16989306e-07,
                1.96474578e-07,
                1.26548863e-07,
                -2.15131614e-07,
                -8.42023089e-07,
                -1.69460748e-06,
                -2.57692109e-06,
                -3.34278386e-06,
                -4.01888775e-06,
                -4.61567113e-06,
                -5.06517546e-06,
                -5.32519134e-06,
                -5.62298298e-06,
                -6.32772252e-06,
                -7.09340375e-06,
                -7.83084282e-06,
                -8.37073198e-06,
                -8.61748653e-06,
            ]
        ),
        dims=["z"],
    ),
    "Q2": xr.DataArray(
        np.array(
            [
                5.79215886e-15,
                2.24957400e-15,
                1.12081588e-14,
                9.89009054e-15,
                4.54772839e-15,
                9.43312857e-15,
                1.34982100e-14,
                -1.24677361e-14,
                1.36172003e-14,
                1.92486697e-14,
                1.44116890e-14,
                2.28235185e-14,
                5.00025836e-14,
                9.49776025e-14,
                1.15005967e-13,
                6.97802600e-14,
                2.97835260e-14,
                -1.30904152e-13,
                -7.72489721e-13,
                -1.97649028e-12,
                -4.11465008e-12,
                -7.38211201e-12,
                -1.25041159e-11,
                -1.98480097e-11,
                -3.13915588e-11,
                -4.89549131e-11,
                -7.08426198e-11,
                -9.24095626e-11,
                -1.09408735e-10,
                -1.23621432e-10,
                -1.40352790e-10,
                -1.58508318e-10,
                -1.72290028e-10,
                -1.81769891e-10,
                -1.90908803e-10,
                -1.98951125e-10,
                -2.08526538e-10,
                -2.14533823e-10,
                -2.19329868e-10,
                -2.29240937e-10,
                -2.34127631e-10,
                -2.35384411e-10,
                -2.47300217e-10,
                -2.66732082e-10,
                -2.74357467e-10,
                -2.84335649e-10,
                -2.63560456e-10,
                -2.09654934e-10,
                -2.01830973e-10,
                -2.21732193e-10,
                -2.80288179e-10,
                -3.52572350e-10,
                -4.23294830e-10,
                -4.31227746e-10,
                -3.31221694e-10,
                -3.33124444e-10,
                -2.94564091e-10,
                -2.51299539e-10,
                -1.91766665e-10,
                -1.26284365e-10,
                -7.75682057e-11,
                -5.14803733e-11,
                -2.50285726e-11,
                2.74621120e-11,
                5.00190271e-11,
                -5.31839192e-11,
                -2.13862831e-10,
                -2.45614612e-10,
                -2.62215527e-10,
                -2.85726061e-10,
                -2.28477211e-10,
                -1.82936211e-10,
                -1.82102170e-10,
                -1.84120924e-10,
                -1.71822074e-10,
                -1.55057481e-10,
                -1.39221230e-10,
                -1.42516913e-10,
                -1.13287559e-10,
            ]
        ),
        dims=["z"],
    ),
}
# account for possible dQ1 and dQ2 naming
OFFLINE_BIASES["dQ1"] = OFFLINE_BIASES["Q1"]
OFFLINE_BIASES["dQ2"] = OFFLINE_BIASES["Q2"]


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Path to a model to-be-loaded.
        variables: Mapping from state names to name of corresponding tendency predicted
            by model. For example: {"air_temperature": "dQ1"}.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
        bias_correction_factor: if provided, add this factor times the hard-coded bias
            for each given tendency name. For example: {"Q1": -1, "Q2": -1.5}.
        scale_factor: if provided, multiply given tendency by this number. For example:
            {"Q1": 1.1, "Q2": 0.95}.
    """

    url: str
    variables: Mapping[str, str]
    limit_negative_humidity: bool = True
    online: bool = True
    bias_correction_factor: Mapping[str, float] = dataclasses.field(
        default_factory=dict
    )
    scale_factor: Mapping[str, float] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        self.model = fv3fit.load(self.config.url)

    def predict(self, inputs: State) -> State:
        tendencies = self.model.predict(xr.Dataset(inputs))
        for name, factor in self.config.bias_correction_factor.items():
            nz = min(tendencies.sizes["z"], len(OFFLINE_BIASES[name]))
            tendencies[name] += factor * OFFLINE_BIASES[name][:nz]
        for name, factor in self.config.scale_factor.items():
            tendencies[name] *= factor
        if self.config.limit_negative_humidity:
            dQ1, dQ2 = non_negative_sphum(
                inputs[SPHUM], tendencies["dQ1"], tendencies["dQ2"], self.timestep
            )
            tendencies.update({"dQ1": dQ1, "dQ2": dQ2})
        state_prediction: State = {}
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                state_prediction[variable_name] = (
                    inputs[variable_name] + tendencies[tendency_name] * self.timestep
                )
        return state_prediction

    def apply(self, prediction: State, state: State):
        if self.config.online:
            state.update(prediction)

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        return self.model.input_variables
