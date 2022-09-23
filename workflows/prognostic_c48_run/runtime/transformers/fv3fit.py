from collections import defaultdict
import dataclasses
from typing import Mapping, MutableMapping, Iterable, Hashable, Sequence, Tuple

import numpy as np
import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import (
    non_negative_sphum_mse_conserving,
    MultiModelAdapter,
)
from runtime.types import State, Diagnostics
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
        diagnostics: sequence of names for model predictions that should be saved as
            diagnostics. May overlap with tendency or state predictions.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Sequence[str]
    tendency_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    state_predictions: Mapping[str, str] = dataclasses.field(default_factory=dict)
    diagnostics: Sequence[str] = dataclasses.field(default_factory=list)
    limit_negative_humidity: bool = True
    online: bool = True

    def __post_init__(self):
        state_targets = list(self.state_predictions.values())
        tendency_targets = list(self.tendency_predictions.values())
        if len(set(state_targets)) < len(state_targets):
            raise ValueError(
                "Cannot have multiple state predictions for same variable."
            )
        if len(set(state_targets).intersection(tendency_targets)) > 0:
            raise ValueError(
                "A variable cannot be updated by tendency and state predictions."
            )


def merge_masked(left: State, right: State) -> State:
    merged = {}
    for key, left_value in left.items():
        try:
            right_value = right[key]
        except KeyError:
            merged[key] = left[key]
        else:
            merged[key] = left_value.where(left_value.notnull(), right_value)
    return merged


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        models = [fv3fit.load(url) for url in self.config.url]
        self.model = MultiModelAdapter(models)  # type: ignore
        self.tendency_names = defaultdict(list)
        for k, v in self.config.tendency_predictions.items():
            self.tendency_names[v].append(k)
        self.state_names = {v: k for k, v in self.config.state_predictions.items()}

    def predict(self, inputs: State) -> Tuple[State, Diagnostics]:
        prediction = self.model.predict(xr.Dataset(inputs))
        tendencies = {
            k: sum([prediction[item] for item in v])
            for k, v in self.tendency_names.items()
        }
        state_updates: MutableMapping[Hashable, xr.DataArray] = {
            k: prediction[v] for k, v in self.state_names.items()
        }
        diagnostics: Diagnostics = {k: prediction[k] for k in self.config.diagnostics}

        if self.config.limit_negative_humidity:
            limited_tendencies = self.non_negative_sphum_limiter(tendencies, inputs)
            tendencies.update(limited_tendencies)

        for name in tendencies:
            with xr.set_options(keep_attrs=True):
                state_updates[name] = inputs[name] + tendencies[name] * self.timestep
        return state_updates, diagnostics

    def apply(self, prediction: State, state: State):
        if self.config.online:
            state.update(merge_masked(prediction, state))

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        return list(set(self.model.input_variables) | set(self.tendency_names))

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
        q2_new = q2_new.where(tendencies[SPHUM].notnull(), other=np.nan)
        limited_tendencies[SPHUM] = q2_new
        if q1_new is not None:
            q1_new = q1_new.where(tendencies[TEMP].notnull(), other=np.nan)
            limited_tendencies[TEMP] = q1_new
        return limited_tendencies
