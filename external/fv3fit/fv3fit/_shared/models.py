from typing import Iterable, Set, Hashable, Sequence
import fsspec
import yaml
import os
import xarray as xr
import vcm
import dataclasses

from . import io
from .predictor import Predictor
from .config import (
    DerivedModelHyperparameters,
    register_training_function,
    get_training_function,
)


@register_training_function("DerivedModel", DerivedModelHyperparameters)
def train_derived_model(
    input_variables: Iterable[str],
    output_variables: Iterable[str],
    hyperparameters: DerivedModelHyperparameters,
    train_batches: Sequence[xr.Dataset],
    validation_batches: Sequence[xr.Dataset],
):

    model = DerivedModel(
        "sample",
        base_model_input_variables=input_variables,
        output_variables=output_variables,
        hyperparameters=hyperparameters,
    )
    # TODO: make use of validation_batches, currently validation dataset is
    # passed through hyperparameters.fit_kwargs
    model.fit(train_batches, validation_batches)
    return model


@io.register("derived_model")
class DerivedModel(Predictor):
    _BASE_MODEL_DIRECTORY = "base_model_data"
    _HYPERPARAMETERS_FILENAME = "hyperparameters.yml"
    _OPTIONS_FILENAME = "options.yml"

    def __init__(
        self,
        sample_dim_name: str,
        base_model_input_variables: Iterable[str],
        output_variables: Iterable[str],
        hyperparameters: DerivedModelHyperparameters,
    ):
        """

        Args:
            sample_dim_name: name of stacked sample dim
            base_model_input_variables: input variables for baseML model
                predictions, *should not* include inputs needed for derived
                prediction if they are not ML features. These additional
                inputs should be specified in
                DerivedModelHyperparameters.additional_inputs
            output_variables: output variables of the base ML model
                i.e. does not include the derived prediction variables, these
                are specified in DerivedModelHyperparameters.derived_variables
            hyperparameters: DerivedModelHyperparameters class
        


        """

        full_input_variables = sorted(
            list(set(base_model_input_variables + hyperparameters.additional_inputs))
        )

        # DerivedModel.input_variables (what the prognostic run uses to grab
        # necessary state for input to .predict()) is the set of
        # base_model_input_variables arg and hyperparameters.additional_inputs.
        super().__init__(sample_dim_name, full_input_variables, output_variables)
        self._hyperparameters = hyperparameters
        self._base_model_type = hyperparameters.base_model_type
        self._base_model_input_variables = base_model_input_variables
        self._base_model_hyperparameters = hyperparameters.base_hyperparameters
        self._derived_variables = hyperparameters.derived_variables

        self._base_model = None

    def fit(
        self, batches: Sequence[xr.Dataset], validation_batches: Sequence[xr.Dataset]
    ):
        # calls the training function for the underlying ML model
        base_training_function = get_training_function(self._base_model_type)
        base_model = base_training_function(
            self._base_model_input_variables,
            self._output_variables,
            self._base_model_hyperparameters,
            batches,
            validation_batches,
        )
        self._base_model = base_model

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        base_prediction = self._base_model.predict(X)
        # derived prediction variables may need additional inputs to compute
        derived_mapping = vcm.DerivedMapping(xr.merge([X, base_prediction]))
        derived_prediction = derived_mapping.dataset(self._derived_variables)
        return xr.merge([base_prediction, derived_prediction])

    def dump(self, path: str):
        self._base_model.dump(os.path.join(path, self._BASE_MODEL_DIRECTORY))
        with fsspec.open(os.path.join(path, self._HYPERPARAMETERS_FILENAME), "w") as f:
            hyperparameters = dataclasses.asdict(self._hyperparameters)
            yaml.safe_dump(hyperparameters, f)
        options = {
            "sample_dim_name": self._sample_dim_name,
            "base_model_input_variables": self._base_model_input_variables,
            "output_variables": self._output_variables,
        }
        with fsspec.open(os.path.join(path, self._OPTIONS_FILENAME), "w") as f:
            yaml.safe_dump(options, f)

    @classmethod
    def load(cls, path: str) -> "DerivedModel":
        with fsspec.open(os.path.join(path, cls._OPTIONS_FILENAME), "r") as f:
            options = yaml.safe_load(f)
        with fsspec.open(os.path.join(path, cls._HYPERPARAMETERS_FILENAME), "r") as f:
            hyperparameters_dict = yaml.safe_load(f)
            hyperparameters = DerivedModelHyperparameters(hyperparameters_dict)

        derived_model = cls(**options, hyperparameters=hyperparameters,)
        derived_model._base_model = io.load(
            os.path.join(path, cls._BASE_MODEL_DIRECTORY)
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
