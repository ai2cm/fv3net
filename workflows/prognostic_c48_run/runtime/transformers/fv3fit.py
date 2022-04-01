import dataclasses
from typing import Mapping, Iterable, Hashable, Sequence

import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import (
    MultiModelAdapter,
    non_negative_sphum_mse_conserving,
)
from runtime.types import State
from runtime.names import SPHUM, TEMP
from runtime.diagnostics.compute import precipitation_accumulation

__all__ = ["Config", "Adapter"]


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Sequence of paths to models that can be loaded with fv3fit.load.
        tendency_predictions: Mapping from state names to name of corresponding tendency
            predicted by ML model. For example: {"air_temperature": "Q1"}.
        state_predictions: Mapping from state names to name of corresponding state
            predicted by ML model. For example:
            {"surface_precipitation_rate": "implied_surface_precipitation_rate"}.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Sequence[str]
    tendency_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    state_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    limit_negative_humidity: bool = True
    online: bool = True


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        models = [fv3fit.load(url) for url in self.config.url]
        self.model = MultiModelAdapter(models)  # type: ignore

    def predict(self, inputs: State) -> State:
        prediction = self.model.predict(xr.Dataset(inputs))
        if self.config.limit_negative_humidity:
            limited_prediction = self.non_negative_sphum_limiter(prediction, inputs)
            prediction = prediction.update(limited_prediction)

        state_update: State = {}
        for variable_name, tendency_name in self.config.tendency_predictions.items():
            with xr.set_options(keep_attrs=True):
                state_update[variable_name] = (
                    inputs[variable_name] + prediction[tendency_name] * self.timestep
                )
        for variable_name, prediction_name in self.config.state_predictions.items():
            state_update[variable_name] = prediction[prediction_name]
        print(state_update)
        return state_update

    def apply(self, prediction: State, state: State):
        if self.config.online:
            _replace_precip_rate_with_accumulation(prediction, self.timestep)
            state.update(prediction)

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        return list(
            set(self.model.input_variables)
            | set(self.config.tendency_predictions)
            | set(self.config.state_predictions)
        )

    def non_negative_sphum_limiter(self, tendencies, inputs):
        limited_tendencies = {}
        if SPHUM not in self.config.tendency_predictions:
            raise NotImplementedError(
                "Cannot limit specific humidity tendencies if specific humidity "
                "updates not being predicted."
            )
        q2_name = self.config.tendency_predictions[SPHUM]
        q1_name = self.config.tendency_predictions.get(TEMP)
        q2_new, q1_new = non_negative_sphum_mse_conserving(
            inputs[SPHUM],
            tendencies[q2_name],
            self.timestep,
            q1=tendencies.get(q1_name),
        )
        limited_tendencies[q2_name] = q2_new
        if q1_name is not None:
            limited_tendencies[q1_name] = q1_new
        return limited_tendencies


TOTAL_PRECIP_RATE = "total_precipitation_rate"
TOTAL_PRECIP = "total_precipitation"  # has units of m


# TODO: define setter for 'surface_precipitation_rate' in fv3gfs-wrapper
# so that we do not need to do this conversion here. For now, this function
# is copied from loop.py
def _replace_precip_rate_with_accumulation(  # type: ignore
    state_updates: State, dt: float
) -> State:
    # Precipitative ML models predict a rate, but the precipitation to update
    # in the state is an accumulated total over the timestep
    if TOTAL_PRECIP_RATE in state_updates:
        state_updates[TOTAL_PRECIP] = precipitation_accumulation(
            state_updates[TOTAL_PRECIP_RATE], dt
        )
        state_updates.pop(TOTAL_PRECIP_RATE)
