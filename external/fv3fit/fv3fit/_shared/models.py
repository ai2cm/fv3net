from typing import Iterable, Set, Hashable, List
import fsspec
import yaml
import os
import numpy as np
import xarray as xr
import vcm

from . import io
from .predictor import Predictor


@io.register("derived_model")
class DerivedModel(Predictor):
    _CONFIG_FILENAME = "derived_model.yaml"
    _BASE_MODEL_SUBDIR = "base_model_data"

    def __init__(
        self, model: Predictor, derived_output_variables: List[Hashable],
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
        # if base_model is itself a DerivedModel, combine the underlying base model
        # and combine derived attributes instead of wrapping twice with DerivedModel
        if isinstance(model, DerivedModel):
            self.base_model: Predictor = model.base_model
            existing_derived_outputs = model._derived_output_variables  # type: ignore
            self._derived_output_variables = (
                existing_derived_outputs + derived_output_variables
            )

        else:
            self.base_model = model
            self._derived_output_variables = derived_output_variables
        self._additional_input_variables = self.get_additional_inputs()

        full_input_variables = sorted(
            list(
                set(
                    list(model.input_variables) + list(self._additional_input_variables)
                )
            )
        )
        full_output_variables = sorted(
            list(set(list(model.output_variables) + list(derived_output_variables)))
        )
        self._check_derived_predictions_supported()
        # DerivedModel.input_variables (what the prognostic run uses to grab
        # necessary state for input to .predict()) is the set of
        # base_model_input_variables arg and hyperparameters.additional_inputs.
        super().__init__(full_input_variables, full_output_variables)

    def get_additional_inputs(self):
        derived_variable_inputs = vcm.DerivedMapping.find_all_required_inputs(
            self._derived_output_variables
        )
        return [
            input
            for input in derived_variable_inputs
            if input not in self.base_model.output_variables
        ]

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        self._check_additional_inputs_present(X)
        base_prediction = self.base_model.predict(X)
        print("\n ... \n base prediction: \n ")
        print(base_prediction, "\n ...")
        required_inputs = vcm.safe.get_variables(X, self._additional_input_variables)
        derived_mapping = vcm.DerivedMapping(
            xr.merge([required_inputs, base_prediction])
        )
        derived_prediction = derived_mapping.dataset(self._derived_output_variables)
        return xr.merge([base_prediction, derived_prediction])

    def dump(self, path: str):
        base_model_path = os.path.join(path, self._BASE_MODEL_SUBDIR)
        options = {
            "derived_output_variables": self._derived_output_variables,
            "model": base_model_path,
        }
        io.dump(self.base_model, base_model_path)
        with fsspec.open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
            yaml.safe_dump(options, f)

    @classmethod
    def load(cls, path: str) -> "DerivedModel":
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        base_model = io.load(config["model"])
        derived_output_variables = config["derived_output_variables"]
        derived_model = cls(base_model, derived_output_variables)
        return derived_model

    def _check_additional_inputs_present(self, X: xr.Dataset):
        missing_additional_inputs = np.setdiff1d(
            self._additional_input_variables, list(X.data_vars)
        )
        if len(missing_additional_inputs) > 0:
            raise KeyError(
                f"Missing additional inputs {missing_additional_inputs} in "
                "input dataset needed to compute derived prediction variables. "
                "Make sure these are present in the data and included in the "
                "DerivedModel config under additional_input_variables."
            )

    def _check_derived_predictions_supported(self):
        invalid_derived_variables = np.setdiff1d(
            self._derived_output_variables, list(vcm.DerivedMapping.VARIABLES)
        )
        if len(invalid_derived_variables) > 0:
            raise ValueError(
                f"Invalid variables {invalid_derived_variables} "
                "provided in init arg derived_output_variables. "
                "Variables in this arg must be available as derived variables "
                "in vcm.DerivedMapping."
            )


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
        outputs = set(self._models[0].output_variables)
        for model in self._models:
            if set(model.output_variables) != outputs:
                raise ValueError(
                    "all models in ensemble must have same outputs, "
                    f"got {outputs} and {set(model.output_variables)}"
                )
            input_variables.update(model.input_variables)
            output_variables.update(model.output_variables)
        super().__init__(
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
