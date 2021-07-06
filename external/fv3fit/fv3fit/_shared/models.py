from typing import Iterable, Set, Hashable
import fsspec
import yaml
import os
import xarray as xr
import vcm

from . import io
from .predictor import Predictor


@io.register("derived_model")
class DerivedModel(Predictor):
    _CONFIG_FILENAME = "derived_model.yaml"
    _BASE_MODEL_DIRECTORY = "base_model_data"

    def __init__(
        self,
        base_model: Predictor,
        additional_input_variables: Iterable[str],
        derived_output_variables: Iterable[str],
    ):
        """

        Args:
            base_model: trained ML model whose predicted output(s) will be
                used to derived the additional derived_output_variables.
            additional_input_variables: inputs needed for derived
                prediction if they are not ML features.
            derived_output_variables: derived prediction variables that are NOT
                part of the set of base_model.output_variables. Should
                correspond to variables available through vcm.DerivedMapping.
        """
        self._base_model = base_model
        self._additional_input_variables = additional_input_variables
        self._derived_output_variables = derived_output_variables

        sample_dim_name = base_model.sample_dim_name

        full_input_variables = sorted(
            list(set(base_model.input_variables + additional_input_variables))
        )
        full_output_variables = sorted(
            list(set(base_model.output_variables + derived_output_variables))
        )

        # DerivedModel.input_variables (what the prognostic run uses to grab
        # necessary state for input to .predict()) is the set of
        # base_model_input_variables arg and hyperparameters.additional_inputs.
        super().__init__(sample_dim_name, full_input_variables, full_output_variables)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        base_prediction = self._base_model.predict(X)
        # derived prediction variables may need additional inputs to compute
        derived_mapping = vcm.DerivedMapping(xr.merge([X, base_prediction]))
        derived_prediction = derived_mapping.dataset(self._derived_variables)
        return xr.merge([base_prediction, derived_prediction])

    def dump(self, path: str):
        raise NotImplementedError(
            "no dump method yet for this class, you can define one manually "
            "using instructions at "
            "http://vulcanclimatemodeling.com/docs/fv3fit/derived_model.html"
        )

    @classmethod
    def load(cls, path: str) -> "DerivedModel":
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        base_model = io.load(config["model"])
        additional_input_variables = config["additional_input_variables"]
        derived_output_variables = config["derived_output_variables"]
        derived_model = cls(
            base_model, additional_input_variables, derived_output_variables
        )
        return derived_model


@io.register("ensemble")
class EnsembleModel(Predictor):

    _CONFIG_FILENAME = "ensemble_model.yaml"

    def __init__(self, models: Iterable[Predictor], reduction: str):
        self._models = tuple(models)
        if len(self._models) == 0:
            raise ValueError("at least one model must be given")
        if reduction.lower() not in ("mean", "median"):
            raise NotImplementedError(
                f"Got reduction {reduction}: only mean, median supported"
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
            input_variables=tuple(sorted(input_variables)),
            output_variables=tuple(sorted(output_variables)),
        )

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset from an input xarray dataset."""
        outputs = [m.predict(X) for m in self._models]
        ds = xr.concat(outputs, dim="member")
        if self._reduction == "median":
            return ds.median(dim="member")
        else:
            return ds.mean(dim="member")

    def dump(self, path):
        raise NotImplementedError(
            "no dump method yet for this class, you can define one manually "
            "using instructions at "
            "http://vulcanclimatemodeling.com/docs/fv3fit/ensembles.html"
        )

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load a serialized model from a directory."""
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        models = [io.load(path) for path in config["models"]]
        reduction = config["reduction"]
        return cls(models, reduction)
