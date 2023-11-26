"""Code for machine Learning in prognostic runs
"""
import dataclasses
import logging
import os
from typing import Hashable, Iterable, Mapping, Optional, Sequence, Set, cast

import fv3fit
import xarray as xr
from runtime.diagnostics import (
    compute_diagnostics,
    compute_ml_momentum_diagnostics,
    enforce_heating_and_moistening_tendency_constraints,
)
from runtime.names import is_state_update_variable, is_tendency_variable
from runtime.types import Diagnostics, State

import vcm


__all__ = ["MachineLearningConfig", "PureMLStepper", "open_model"]


logger = logging.getLogger(__name__)

NameDict = Mapping[Hashable, Hashable]


@dataclasses.dataclass
class MachineLearningConfig:
    """Machine learning configurations

    Attributes:
        model: list of URLs to fv3fit models.
        diagnostic_ml: do not apply ML tendencies if true.
        input_standard_names: mapping from non-standard names to the standard
            ones used by the model. Renames the ML inputs. Useful if the
            input variables in the ML model are inconsistent with
            the canonical names used in the wrapper.
        output_standard_names: mapping from non-standard names to the standard
            ones used by the model. Renames the ML predictions.
        use_mse_conserving_humidity_limiter: if true, use an MSE-conserving
            humidity limiter. If false, use a previous method that did not
            conserve MSE. This option is available for backwards compatibility.
        scaling: if given, scale the outputs by the given factor. This is a manually
            defined alteration of the model, and should not be used for
            normalization.

    Example::

        MachineLearningConfig(
            model=["gs://vcm-ml-scratch/test-annak/ml-pipeline-output"],
            diagnostic_ml=False,
            input_standard_names={},
            output_standard_names={},
        )

    """

    model: Sequence[str] = dataclasses.field(default_factory=list)
    diagnostic_ml: bool = False
    input_standard_names: Mapping[Hashable, Hashable] = dataclasses.field(
        default_factory=dict
    )
    output_standard_names: Mapping[Hashable, Hashable] = dataclasses.field(
        default_factory=dict
    )
    use_mse_conserving_humidity_limiter: bool = True
    scaling: Mapping[str, float] = dataclasses.field(default_factory=dict)


def invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


def rename_dataset_members(ds: xr.Dataset, rename: NameDict) -> xr.Dataset:
    all_names = set(ds.dims) & set(rename)
    rename_restricted = {key: rename[key] for key in all_names}
    redimed = ds.rename_dims(rename_restricted)

    all_names = set(ds.data_vars) & set(rename)
    rename_restricted = {key: rename[key] for key in all_names}
    return redimed.rename(rename_restricted)


class RenamingAdapter:
    """Adapter object for renaming model variables

    Attributes:
        model: a model to rename
        rename_in: mapping from standard names to input names of model
        rename_out: mapping from standard names to the output names of model

    """

    def __init__(
        self, model: fv3fit.Predictor, rename_in: NameDict, rename_out: NameDict = None
    ):
        self.model = model
        self.rename_in = rename_in
        self.rename_out = {} if rename_out is None else rename_out

    def _rename_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return rename_dataset_members(ds, self.rename_in)

    def _rename_outputs(self, ds: xr.Dataset) -> xr.Dataset:
        return rename_dataset_members(ds, invert_dict(self.rename_out))

    @property
    def input_variables(self) -> Set[str]:
        invert_rename_in = invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_variables}

    def predict(self, arg: xr.Dataset) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_)
        return self._rename_outputs(prediction)


class MultiModelAdapter:
    def __init__(
        self,
        models: Iterable[RenamingAdapter],
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
        return {var for model_vars in vars for var in model_vars}

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
        rename_in = config.input_standard_names
        rename_out = config.output_standard_names
        models.append(RenamingAdapter(model, rename_in, rename_out))
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


class PureMLStepper:
    label = "machine_learning"

    def __init__(
        self,
        model: MultiModelAdapter,
        timestep: float,
        hydrostatic: bool,
        mse_conserving_limiter: bool = True,
    ):
        """A stepper for predicting machine learning tendencies and state updates.

        Args:
            model: the machine learning model.
            timestep: physics timestep in seconds.
            hydrostatic: whether simulation is hydrostatic. For net heating diagnostic.
            mse_conserving_limiter (optional): whether to use MSE-conserving humidity
                limiter. Defaults to True.
        """
        self.model = model
        self.timestep = timestep
        self.hydrostatic = hydrostatic
        self.mse_conserving_limiter = mse_conserving_limiter

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}
        prediction: State = predict(self.model, state)

        tendency, state_updates = {}, {}
        for key, value in prediction.items():
            if is_state_update_variable(key, state):
                state_updates[key] = value
            elif is_tendency_variable(key):
                tendency[key] = value
            else:
                diagnostics[key] = value

        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]

        # Adjust dQ1 and dQ2 to ensure non-negative humidity and optionally conserve MSE
        # and add diagnostics tracking the tendency adjustments related to this step
        (
            tendency_updates,
            diagnostics_updates,
        ) = enforce_heating_and_moistening_tendency_constraints(
            state=state,
            tendency=tendency,
            timestep=self.timestep,
            mse_conserving=self.mse_conserving_limiter,
            hydrostatic=self.hydrostatic,
            temperature_tendency_name="dQ1",
            humidity_tendency_name="dQ2",
        )
        tendency.update(tendency_updates)
        diagnostics.update(diagnostics_updates)

        return (
            tendency,
            diagnostics,
            state_updates,
        )

    def get_diagnostics(self, state, tendency):
        diags = compute_diagnostics(state, tendency, self.label, self.hydrostatic)
        momentum_diags = compute_ml_momentum_diagnostics(state, tendency)
        diags.update(momentum_diags)
        return diags, diags[f"net_moistening_due_to_{self.label}"]
