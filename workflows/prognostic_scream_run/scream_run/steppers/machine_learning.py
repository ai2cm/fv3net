import dataclasses
from typing import (
    Hashable,
    Iterable,
    MutableMapping,
    Mapping,
    Optional,
    Sequence,
    Set,
    cast,
)
import fv3fit
import xarray as xr
import vcm
import os


State = MutableMapping[Hashable, xr.DataArray]


@dataclasses.dataclass
class MachineLearningConfig:
    """Machine learning configurations

    Attributes:
        model: list of URLs to fv3fit models.
        diagnostic_ml: do not apply ML tendencies if true.
        scaling: if given, scale the outputs by the given factor. This is a manually
            defined alteration of the model, and should not be used for
            normalization.
    Example::

        MachineLearningConfig(
            model=["gs://vcm-ml-scratch/test-annak/ml-pipeline-output"],
            diagnostic_ml=False,
        )

    """

    model: Sequence[str] = dataclasses.field(default_factory=list)
    diagnostic_ml: bool = False
    scaling: Mapping[str, float] = dataclasses.field(default_factory=dict)


class MultiModelAdapter:
    def __init__(
        self,
        models: Iterable[fv3fit.Predictor],
        scaling: Optional[Mapping[str, float]] = None,
    ):
        """
        Args:
            models: models for which to combine predictions
            scaling: if given, scale the predictions by the given factor
        """
        self.models = models
        if scaling is None:
            self._scaling: Mapping[str, float] = {}
        else:
            self._scaling = scaling

    @property
    def input_variables(self) -> Set[str]:
        vars = [model.input_variables for model in self.models]
        return {var for model_vars in vars for var in model_vars}  # type: ignore

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(arg))
        ds = xr.merge(predictions)
        for var, scale in self._scaling.items():
            ds[var] *= scale
        return ds


def open_model(config: MachineLearningConfig) -> MultiModelAdapter:
    model_paths = config.model
    models = []
    for path in model_paths:
        model = cast(fv3fit.Predictor, fv3fit.load(path))
        models.append(model)
    return MultiModelAdapter(models, scaling=config.scaling)


def download_model(config: MachineLearningConfig, path: str) -> Sequence[str]:
    """Download models to local path and return the local paths"""
    remote_model_paths = config.model
    local_model_paths = []
    for i, remote_path in enumerate(remote_model_paths):
        local_path = os.path.join(path, str(i))
        os.makedirs(local_path)
        fs = vcm.cloud.get_fs(remote_path)
        fs.get(remote_path, local_path, recursive=True)
        local_model_paths.append(local_path)
    return local_model_paths


def predict(model: MultiModelAdapter, state: State) -> State:
    """Given ML model and state, return prediction"""
    state_loaded = {key: state[key] for key in model.input_variables}
    ds = xr.Dataset(state_loaded)  # type: ignore
    output = model.predict(ds)
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}
