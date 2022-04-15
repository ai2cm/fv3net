import dataclasses
from typing import Mapping, MutableMapping, Iterable, Hashable, Sequence, Literal

import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import (
    non_negative_sphum_mse_conserving,
    MultiModelAdapter,
)
from runtime.types import State
from runtime.names import SPHUM, TEMP

__all__ = ["Config", "Adapter"]


@dataclasses.dataclass
class MLOutputApplier:
    """Configuration of how to apply ML predictions to state.

    Attrs:
        target_name: Name of the variable to apply ML predictions to.
        method: How to apply ML predictions to the target variable.
    """

    target_name: str
    method: Literal["tendency", "state"]

    def apply(
        self, inputs: State, prediction: xr.DataArray, timestep: float
    ) -> xr.DataArray:
        if self.method == "tendency":
            with xr.set_options(keep_attrs=True):
                output = inputs[self.target_name] + prediction * timestep
        elif self.method == "state":
            output = prediction
        return output


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Sequence of paths to models that can be loaded with fv3fit.load.
        output_targets: Mapping from names of outputs predicted by ML model to
            MLOutputApplier configuration. For example:
            {"Q1": {"target_name": "air_temperature", "method": "tendency"}.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Sequence[str]
    output_targets: Mapping[str, MLOutputApplier]
    limit_negative_humidity: bool = True
    online: bool = True


class PredictionInverter:
    """Take ML predictions and convert to mapping keyed on state variable names"""

    def __init__(self, targets: Mapping[str, MLOutputApplier]):
        states = [t.target_name for t in targets.values() if t.method == "state"]
        tendencies = [t.target_name for t in targets.values() if t.method == "tendency"]
        if len(set(states)) < len(states):
            raise ValueError(
                "Cannot have multiple state predictions for same variable."
            )
        if len(set(states).intersection(tendencies)) > 0:
            raise ValueError(
                "A variable cannot be updated by tendency and state predictions."
            )
        self.targets = targets

    def invert(self, predictions: State) -> State:
        """Separate a dataset of predictions into tendency and state predictions.

        Args:
            predictions: tendency and/or state predictions, keyed on ML output names.

        Returns:
            Mapping of tendencies and state updates keyed on state names to be updated.

        Note:
            Tendencies are summed over all predictions for a given variable.
        """
        output: MutableMapping[Hashable, xr.DataArray] = {}
        for ml_output_name, target in self.targets.items():
            if target.method == "tendency":
                if target.target_name in output:
                    output[target.target_name] += predictions[ml_output_name]
                else:
                    output[target.target_name] = predictions[ml_output_name]
            elif target.method == "state":
                output[target.target_name] = predictions[ml_output_name]
        return output

    def apply(self, inputs: State, predictions: State, timestep: float) -> State:
        """Apply tendency and state updates to the input state."""
        apply_methods = {t.target_name: t.apply for t in self.targets.values()}
        output: MutableMapping[Hashable, xr.DataArray] = {}
        for name, apply in apply_methods.items():
            output[name] = apply(inputs, predictions[name], timestep)
        return output


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        models = [fv3fit.load(url) for url in self.config.url]
        self.model = MultiModelAdapter(models)  # type: ignore
        self.inverter = PredictionInverter(self.config.output_targets)

    def predict(self, inputs: State) -> State:
        prediction = self.model.predict(xr.Dataset(inputs))
        tendencies = self.inverter.invert(prediction)
        if self.config.limit_negative_humidity:
            limited_tendencies = self.non_negative_sphum_limiter(tendencies, inputs)
            tendencies.update(limited_tendencies)
        return self.inverter.apply(inputs, tendencies, self.timestep)

    def apply(self, prediction: State, state: State):
        if self.config.online:
            state.update(prediction)

    def partial_fit(self, inputs: State, state: State):
        pass

    @property
    def input_variables(self) -> Iterable[Hashable]:
        tendency_target_names = [
            t.target_name
            for t in self.config.output_targets.values()
            if t.method == "tendency"
        ]
        return list(set(self.model.input_variables) | set(tendency_target_names))

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
