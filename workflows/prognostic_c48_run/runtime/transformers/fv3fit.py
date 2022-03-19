import dataclasses
from typing import Mapping, Iterable, Hashable, Sequence, Union

import xarray as xr
import fv3fit
from runtime.steppers.machine_learning import non_negative_sphum, MultiModelAdapter
from runtime.types import State
from runtime.names import SPHUM

__all__ = ["Config", "Adapter"]


@dataclasses.dataclass
class Config:
    """
    Attributes:
        url: Path to a model to-be-loaded, or sequence of paths.
        variables: Mapping from state names to name of corresponding tendency predicted
            by model. For example: {"air_temperature": "dQ1"}.
        limit_negative_humidity: if True, rescale tendencies to not allow specific
            humidity to become negative.
        online: if True, the ML predictions will be applied to model state.
    """

    url: Union[str, Sequence[str]]
    variables: Mapping[str, str]
    limit_negative_humidity: bool = True
    online: bool = True


@dataclasses.dataclass
class Adapter:
    config: Config
    timestep: float

    def __post_init__(self: "Adapter"):
        if isinstance(self.config.url, str):
            self.model = fv3fit.load(self.config.url)
        else:
            models = [fv3fit.load(url) for url in self.config.url]
            self.model = MultiModelAdapter(models)

    def predict(self, inputs: State) -> State:
        tendencies = self.model.predict(xr.Dataset(inputs))
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
