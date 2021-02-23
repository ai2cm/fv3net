"""Code for machine Learning in prognostic runs
"""
import dataclasses
import logging
from runtime import diagnostics
from typing import Any, Hashable, List, Mapping, Sequence, Set, Iterable, Tuple, cast

import runtime
import xarray as xr
from functools import partial

import fv3fit
from vcm import thermo
from runtime.names import (
    DELP,
    PRECIP_RATE,
    SPHUM,
    TENDENCY_TO_STATE_NAME,
    TOTAL_PRECIP,
)

from runtime.steppers.base import (
    Stepper,
    LoggingMixin,
    apply,
    precipitation_sum,
)
from runtime.types import State, Diagnostics

__all__ = ["MachineLearningConfig", "MLStepper", "open_model"]


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

    Example::

        MachineLearningConfig(
            model=["gs://vcm-ml-data/test-annak/ml-pipeline-output"],
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


def non_negative_sphum(
    sphum: xr.DataArray, dQ1: xr.DataArray, dQ2: xr.DataArray, dt: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    delta = dQ2 * dt
    reduction_ratio = (-sphum) / (dt * dQ2)  # type: ignore
    dQ1_updated = xr.where(sphum + delta >= 0, dQ1, reduction_ratio * dQ1)
    dQ2_updated = xr.where(sphum + delta >= 0, dQ2, reduction_ratio * dQ2)
    return dQ1_updated, dQ2_updated


def _invert_dict(d: Mapping) -> Mapping:
    return dict(zip(d.values(), d.keys()))


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

    def _rename(self, ds: xr.Dataset, rename: NameDict) -> xr.Dataset:

        all_names = set(ds.dims) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        redimed = ds.rename_dims(rename_restricted)

        all_names = set(ds.data_vars) & set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        return redimed.rename(rename_restricted)

    def _rename_inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, self.rename_in)

    def _rename_outputs(self, ds: xr.Dataset) -> xr.Dataset:
        return self._rename(ds, _invert_dict(self.rename_out))

    @property
    def input_variables(self) -> Set[str]:
        invert_rename_in = _invert_dict(self.rename_in)
        return {invert_rename_in.get(var, var) for var in self.model.input_variables}

    def predict_columnwise(self, arg: xr.Dataset, **kwargs) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict_columnwise(input_, **kwargs)
        return self._rename_outputs(prediction)


class MultiModelAdapter:
    def __init__(self, models: Iterable[RenamingAdapter]):
        self.models = models

    @property
    def input_variables(self) -> Set[str]:
        vars = [model.input_variables for model in self.models]
        return {var for model_vars in vars for var in model_vars}

    def predict_columnwise(self, arg: xr.Dataset, **kwargs) -> xr.Dataset:
        predictions = []
        for model in self.models:
            predictions.append(model.predict_columnwise(arg, **kwargs))
        return xr.merge(predictions)


def open_model(config: MachineLearningConfig) -> MultiModelAdapter:
    model_paths = config.model
    models = []
    for path in model_paths:
        model = fv3fit.load(path)
        rename_in = config.input_standard_names
        rename_out = config.output_standard_names
        models.append(RenamingAdapter(model, rename_in, rename_out))
    return MultiModelAdapter(models)


def predict(model: MultiModelAdapter, state: State) -> State:
    """Given ML model and state, return tendency prediction."""
    state_loaded = {key: state[key] for key in model.input_variables}
    ds = xr.Dataset(state_loaded)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


def PureMLStepper(model: Any, state, timestep: float):
    diagnostics: Diagnostics = {}
    delp = state[DELP]

    tendency: State = predict(model, state)

    dQ1_initial = tendency.get("dQ1", xr.zeros_like(state[SPHUM]))
    dQ2_initial = tendency.get("dQ2", xr.zeros_like(state[SPHUM]))

    dQ1_updated, dQ2_updated = non_negative_sphum(
        state[SPHUM], dQ1_initial, dQ2_initial, dt=timestep,
    )

    rank_updated_points = xr.where(dQ2_initial != dQ2_updated, 1, 0)

    if "dQ1" in tendency:
        diag = thermo.column_integrated_heating(dQ1_updated - tendency["dQ1"], delp)
        diagnostics.update(
            {"column_integrated_dQ1_change_non_neg_sphum_constraint": (diag)}
        )
        tendency.update({"dQ1": dQ1_updated})
    if "dQ2" in tendency:
        diag = thermo.mass_integrate(dQ2_updated - tendency["dQ2"], delp, dim="z")
        diag = diag.assign_attrs({"units": "kg/m^2/s"})
        diagnostics.update(
            {"column_integrated_dQ2_change_non_neg_sphum_constraint": (diag)}
        )
        tendency.update({"dQ2": dQ2_updated})
    dycore_tendencies = {k: v for k, v in tendency.items() if k in ["dQ1", "dQ2"]}
    physics_tendencies = {k: v for k, v in tendency.items() if k in ["dQu", "dQv"]}
    return dycore_tendencies, physics_tendencies, diagnostics, rank_updated_points


class MLStepper(Stepper, LoggingMixin):
    def __init__(
        self,
        state,
        comm: Any,
        timestep: float,
        states_to_output: Sequence[str],
        model: MultiModelAdapter,
        diagnostic_only: bool = False,
    ):
        self._state = state
        self.rank: int = comm.rank
        self.comm = comm
        self._do_only_diagnostic_ml = diagnostic_only
        self._timestep = timestep
        self.model = model
        self._states_to_output = states_to_output

        self._tendencies_to_apply_to_dycore_state: State = {}
        self._tendencies_to_apply_to_physics_state: State = {}

    def get_diagnostics(self, state, tendency):
        return runtime.compute_ml_diagnostics(state, tendency)

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        tendency = self._tendencies_to_apply_to_dycore_state
        diagnostics = self.get_diagnostics(self._state, tendency)
        if self._do_only_diagnostic_ml:
            runtime.rename_diagnostics(diagnostics)
        else:
            updated_state = apply(self._state, tendency, dt=self._timestep)
            updated_state[TOTAL_PRECIP] = precipitation_sum(
                self._state[TOTAL_PRECIP], diagnostics["net_moistening"], self._timestep
            )
            diagnostics[TOTAL_PRECIP] = updated_state[TOTAL_PRECIP]
            self._state.update(updated_state)
        return diagnostics

    def _apply_python_to_physics_state(self) -> Diagnostics:
        self._log_debug(f"Apply python tendencies to physics state")
        variables: List[Hashable] = [
            TENDENCY_TO_STATE_NAME["dQu"],
            TENDENCY_TO_STATE_NAME["dQv"],
            DELP,
        ]
        state = {name: self._state[name] for name in variables}
        tendency = self._tendencies_to_apply_to_physics_state
        updated_state: State = apply(state, tendency, dt=self._timestep)
        diagnostics: Diagnostics = runtime.compute_ml_momentum_diagnostics(
            state, tendency
        )
        if self._do_only_diagnostic_ml:
            runtime.rename_diagnostics(diagnostics)
        else:
            self._state.update(updated_state)

        return diagnostics

    def _compute_python_tendency(self) -> Diagnostics:
        self._log_info("Computing ML Tendency")
        (
            self._tendencies_to_apply_to_dycore_state,
            self._tendencies_to_apply_to_physics_state,
            diagnostics,
            rank_updated_points,
        ) = PureMLStepper(self.model, self._state, self._timestep)

        updated_points = self.comm.reduce(rank_updated_points, root=0)
        if self.comm.rank == 0:
            level_updates = {
                i: int(value)
                for i, value in enumerate(updated_points.sum(["x", "y"]).values)
            }
            logger.info(f"specific_humidity_limiter_updates_per_level: {level_updates}")

        return diagnostics
