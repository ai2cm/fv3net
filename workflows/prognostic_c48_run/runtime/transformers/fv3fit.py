import dataclasses
from typing import Mapping, MutableMapping, Iterable, Hashable, Sequence, Tuple

import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import non_negative_sphum_mse_conserving
from runtime.types import State
from runtime.names import SPHUM, TEMP

__all__ = ["Config", "Adapter"]


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Sequence of paths to models that can be loaded with fv3fit.load.
        tendency_predictions: Mapping from names of outputs predicted by ML model to
            state names. For example: {"Q1": "air_temperature"}. These predictions
            will be multiplied by the physics timestep before being added to the state.
        state_predictions: Mapping from names of outputs predicted by ML model to
            state names. For example:
            {"implied_surface_precipitation_rate": "surface_precipitation_rate"}. The
            state will be set to be equal to these predictions.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Sequence[str]
    tendency_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    state_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    limit_negative_humidity: bool = True
    online: bool = True


class TendencyOrStateMultiModelAdapter:
    def __init__(
        self,
        models: Iterable[fv3fit.Predictor],
        tendency_predictions: Mapping[str, str],
        state_predictions: Mapping[str, str],
    ):
        self.models = models
        if len(set(state_predictions.values())) < len(state_predictions.values()):
            raise ValueError(
                "Cannot have multiple state predictions for same state variable."
            )
        self.tendency_predictions = tendency_predictions
        self.state_predictions = state_predictions

    @property
    def input_variables(self) -> Iterable[Hashable]:
        vars = [model.input_variables for model in self.models]
        return list({var for model_vars in vars for var in model_vars})

    def predict(self, arg: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(arg))
        merged_predictions = xr.merge(predictions)
        tendencies: MutableMapping[str, xr.DataArray] = {}
        state_updates: MutableMapping[str, xr.DataArray] = {}
        for tendency_name, variable_name in self.tendency_predictions.items():
            if variable_name in tendencies:
                tendencies[variable_name] += merged_predictions[tendency_name]
            else:
                tendencies[variable_name] = merged_predictions[tendency_name]
        for prediction_name, variable_name in self.state_predictions.items():
            state_updates[variable_name] = merged_predictions[prediction_name]
        return xr.Dataset(tendencies), xr.Dataset(state_updates)


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        models = [fv3fit.load(url) for url in self.config.url]
        self.model = TendencyOrStateMultiModelAdapter(
            models, self.config.tendency_predictions, self.config.state_predictions
        )

    def predict(self, inputs: State) -> State:
        tendencies, state_updates = self.model.predict(xr.Dataset(inputs))
        if self.config.limit_negative_humidity:
            limited_tendencies = self.non_negative_sphum_limiter(tendencies, inputs)
            tendencies = tendencies.update(limited_tendencies)

        prediction: State = {}
        for name in tendencies:
            with xr.set_options(keep_attrs=True):
                prediction[name] = inputs[name] + tendencies[name] * self.timestep
        for name in state_updates:
            prediction[name] = state_updates[name]
        return prediction

    def apply(self, prediction: State, state: State):
        if self.config.online:
            state.update(prediction)

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        return list(
            set(self.model.input_variables)
            | set(self.config.tendency_predictions.values())
        )

    def non_negative_sphum_limiter(self, tendencies, inputs):
        limited_tendencies = {}
        if SPHUM not in tendencies:
            raise NotImplementedError(
                "Cannot limit specific humidity tendencies if specific humidity "
                "updates not being predicted."
            )
        q2_new, q1_new = non_negative_sphum_mse_conserving(
            inputs[SPHUM], tendencies[SPHUM], self.timestep, q1=tendencies.get(TEMP),
        )
        limited_tendencies[SPHUM] = q2_new
        if q1_new is not None:
            limited_tendencies[TEMP] = q1_new
        return limited_tendencies
