from typing import Iterable
import yaml
import os
import xarray as xr
from . import io
from .predictor import Predictor


@io.register("ensemble")
class EnsembleModel(Predictor):

    _CONFIG_FILENAME = "ensemble_model.yaml"

    def __init__(self, models: Iterable[Predictor], reduction: str):
        self._models = models
        if reduction.lower() not in ("mean", "median"):
            raise NotImplementedError(
                f"only supported reductions are mean and median, got {reduction}"
            )
        self._reduction = reduction

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
