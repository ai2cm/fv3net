import dataclasses
from typing import Callable, Iterable, Optional, Set, Hashable, Sequence, cast, Mapping
import dacite
import fsspec
import yaml
import os
import numpy as np
import xarray as xr
import vcm

from fv3fit._shared.novelty_detector import NoveltyDetector
from fv3fit._shared.taper_function import get_taper_function, taper_mask
from fv3fit._shared.config import TaperConfig

from . import io
from .predictor import Predictor


@io.register("tapered_model")
class TaperedModel(Predictor):
    _CONFIG_FILENAME = "tapered_model.yaml"

    def __init__(self, model, tapering: Mapping[str, TaperConfig]):
        for taper_var in tapering:
            if taper_var not in model.output_variables:
                raise KeyError(
                    f"Tapered variable {taper_var} not in model output variables."
                )
        self.model = model
        self.tapering = tapering

        super().__init__(
            input_variables=tuple(sorted(model.input_variables)),
            output_variables=tuple(sorted(model.output_variables)),
        )

    @classmethod
    def load(cls, path: str) -> "TaperedModel":
        """Load a serialized model from a directory."""
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        model = cast(Predictor, io.load(config["model"]))
        tapering = {
            taper_variable: dacite.from_dict(TaperConfig, taper_config)
            for taper_variable, taper_config in config["tapering"].items()
        }
        return cls(model, tapering)

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        """Predict an output xarray dataset and taper outputs"""
        output = self.model.predict(X)
        for taper_variable, taper_config in self.tapering.items():
            output[taper_variable] = taper_config.apply(output[taper_variable])
        return output

    def dump(self, path):
        raise NotImplementedError(
            "no dump method yet for this class, you can define one manually "
            "using instructions at "
            "http://vulcanclimatemodeling.com/docs/fv3fit/composite-models.html"
        )


@io.register("derived_model")
class DerivedModel(Predictor):
    _CONFIG_FILENAME = "derived_model.yaml"
    _BASE_MODEL_SUBDIR = "base_model_data"

    def __init__(
        self, model: Predictor, derived_output_variables: Sequence[Hashable],
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
        base_model = cast(Predictor, io.load(config["model"]))
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
            "http://vulcanclimatemodeling.com/docs/fv3fit/composite-models.html"
        )

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        """Load a serialized model from a directory."""
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        models = [cast(Predictor, io.load(path)) for path in config["models"]]
        reduction = config["reduction"]
        return cls(models, reduction)


@io.register("output_transformed_model")
class TransformedPredictor(Predictor):
    _CONFIG_FILENAME = "output_transformed_model.yaml"
    _BASE_MODEL_SUBDIR = "base_model_data"

    def __init__(
        self, base_model: Predictor, transforms: Sequence[vcm.DataTransform],
    ):
        """
        Args:
            base_model: trained ML model whose predicted output(s) will be
                used as inputs for the specified transforms.
            transforms: data transformations to apply to model prediction outputs.
        """
        self.base_model = base_model
        self.transforms = transforms
        self.output_transform = vcm.ChainedDataTransform(self.transforms)

        inputs_for_derived = self.output_transform.input_variables
        derived_outputs = self.output_transform.output_variables
        input_variables = set(base_model.input_variables) | set(inputs_for_derived)
        output_variables = set(base_model.output_variables) | set(derived_outputs)

        for name in set(base_model.output_variables):
            input_variables.discard(name)

        super().__init__(sorted(list(input_variables)), sorted(list(output_variables)))

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        prediction = self.base_model.predict(X)
        transform_inputs = xr.merge([prediction, X], compat="override")
        transformed_prediction = self.output_transform.apply(transform_inputs)
        transformed_outputs = transformed_prediction[
            self.output_transform.output_variables
        ]
        return xr.merge([prediction, transformed_outputs])

    def dump(self, path: str):
        base_model_path = os.path.join(path, self._BASE_MODEL_SUBDIR)
        options = {
            "base_model": base_model_path,
            "transforms": [dataclasses.asdict(x) for x in self.transforms],
        }
        io.dump(self.base_model, base_model_path)
        with fsspec.open(os.path.join(path, self._CONFIG_FILENAME), "w") as f:
            yaml.safe_dump(options, f)

    @classmethod
    def load(cls, path: str) -> "TransformedPredictor":
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)
        base_model = cast(
            Predictor, io.load(os.path.join(path, cls._BASE_MODEL_SUBDIR))
        )
        transform_configs = [
            dacite.from_dict(vcm.DataTransform, x) for x in config["transforms"]
        ]
        transformed_model = cls(base_model, transform_configs)
        return transformed_model


@io.register("out_of_sample")
class OutOfSampleModel(Predictor):

    _TAPER_VALUES_OUTPUT_VAR = "taper_values"

    _CONFIG_FILENAME = "out_of_sample_model.yaml"

    def __init__(
        self,
        base_model: Predictor,
        novelty_detector: NoveltyDetector,
        cutoff: float = 0,
        taper: Optional[Callable[[xr.DataArray], xr.DataArray]] = None,
    ):
        """
        Args:
            base_model: trained ML-based predictor, to be used for inputs deemed
                "in-sample" and to be suppressed otherwise
            novelty_detector: trained novelty detector on the same inputs, which
                predicts whether a training sample does _not_ belong the training
                distribution of base_model
            cutoff: a score cutoff for the novelty detector's scores; scores larger
                are deemed out-of-samples, scores smaller are not. Specifying None
                supplies the default cutoff value for the given NoveltyDetector
                implementation
            taper: given an array of novelty scores, determines how much the predicted
                tendencies should be suppressed. Specifying None supplies a default
                "mask" tapering, which either completely suppressed the tendencies
                or does nothing
        """
        self.base_model = base_model
        self.novelty_detector = novelty_detector
        self.cutoff = cutoff
        self.taper = taper or get_taper_function(
            taper_mask.__name__, {"cutoff": cutoff}
        )

        base_inputs = set(base_model.input_variables)
        base_outputs = set(base_model.output_variables)
        novelty_inputs = set(novelty_detector.input_variables)
        novelty_outputs = set(novelty_detector.output_variables)
        taper_output = set([self._TAPER_VALUES_OUTPUT_VAR])
        input_variables = tuple(sorted(base_inputs | novelty_inputs))
        output_variables = tuple(sorted(base_outputs | novelty_outputs | taper_output))
        super().__init__(
            input_variables=input_variables, output_variables=output_variables
        )

    def predict(self, X: xr.Dataset) -> xr.Dataset:
        base_predict = self.base_model.predict(X)
        centered_scores, diagnostics = self.novelty_detector.predict_novelties(
            X, cutoff=self.cutoff
        )
        taper_values = self.taper(centered_scores)
        diagnostics[self._TAPER_VALUES_OUTPUT_VAR] = taper_values

        tapered_predict = xr.Dataset(
            coords=base_predict.coords, attrs=base_predict.attrs
        )
        for output_variable in self.base_model.output_variables:
            tapered_predict[output_variable] = (
                base_predict[output_variable] * taper_values
            )

        return xr.merge([tapered_predict, diagnostics])

    def dump(self, path):
        raise NotImplementedError(
            "no dump method yet for this class, you can define one manually "
            "using instructions at "
            "http://vulcanclimatemodeling.com/docs/fv3fit/composite-models.html"
        )

    @classmethod
    def load(cls, path: str) -> "OutOfSampleModel":
        with fsspec.open(os.path.join(path, cls._CONFIG_FILENAME), "r") as f:
            config = yaml.safe_load(f)

        base_model = cast(Predictor, io.load(config["base_model_path"]))
        novelty_detector = cast(Predictor, io.load(config["novelty_detector_path"]))
        cutoff = config.get("cutoff", 0)

        assert isinstance(novelty_detector, NoveltyDetector)

        default_tapering_config = {
            "name": taper_mask.__name__,
            "cutoff": cutoff,
            "ramp_min": cutoff,
            "ramp_max": 1 if cutoff == 0 else max(cutoff * 2, cutoff / 2),
            "threshold": cutoff,
        }
        tapering_config = {
            **default_tapering_config,
            **config.get("tapering_function", {}),
        }

        taper = get_taper_function(tapering_config["name"], tapering_config)

        model = cls(base_model, novelty_detector, cutoff=cutoff, taper=taper)
        return model
