from typing import Iterable, Set, Hashable
import yaml
import os
import xarray as xr
from . import io
from .predictor import Predictor


@io.register("ensemble")
class EnsembleModel(Predictor):

    _CONFIG_FILENAME = "ensemble_model.yaml"

    def __init__(self, models: Iterable[Predictor], reduction: str):
        self._models = tuple(models)
        if len(self._models) == 0:
            raise ValueError("at least one model must be given")
        if reduction.lower() not in ("mean", "median"):
            raise NotImplementedError(
                f"only supported reductions are mean and median, got {reduction}"
            )
        self._reduction = reduction
        input_variables: Set[Hashable] = set()
        output_variables: Set[Hashable] = set()
        sample_dim_name = self._models[0].sample_dim_name
        outputs = set(self._models[0].output_variables)
        for model in self._models:
            if model.sample_dim_name != sample_dim_name:
                raise ValueError(
                    "all models in ensemble must have same sample_dim_name, "
                    f"got {sample_dim_name} and {model.sample_dim_name}"
                )
            if set(model.output_variables) != outputs:
                raise ValueError(
                    "all models in ensemble must have same outputs, "
                    f"got {outputs} and {set(model.output_variables)}"
                )
            input_variables.update(model.input_variables)
            output_variables.update(model.output_variables)
        super().__init__(
            sample_dim_name,
            input_variables=tuple(input_variables),
            output_variables=tuple(output_variables),
        )

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        outputs = [m.predict(X) for m in self._models]
        ds = xr.concat(outputs, dim="member")
        if self._reduction == "median":
            return ds.median(dim="member")
        else:
            return ds.mean(dim="member")

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load a serialized model from a directory."""
        with open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        models = [io.load(path) for path in config["models"]]
        reduction = config["reduction"]
        return cls(models, reduction)
