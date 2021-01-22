"""Code for machine Learning in prognostic runs
"""
import dataclasses
import logging
from typing import Any, Hashable, List, Mapping, Sequence, Set, Iterable, Tuple, cast

import runtime
import xarray as xr

import fv3fit
from runtime.names import (
    AREA,
    DELP,
    PRECIP_RATE,
    SPHUM,
    TENDENCY_TO_STATE_NAME,
    TOTAL_PRECIP,
)

from runtime.steppers.base import Stepper, LoggingMixin, apply, precipitation_sum
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
            ones used by the model. Renames the ML inputs.
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


def log_updated_tendencies(comm, tendency: State, tendency_updated: State):
    rank_updated_points = xr.where(tendency["dQ2"] != tendency_updated["dQ2"], 1, 0)
    updated_points = comm.reduce(rank_updated_points, root=0)
    if comm.rank == 0:
        level_updates = {
            i: int(value)
            for i, value in enumerate(updated_points.sum(["x", "y"]).values)
        }
        logger.info(f"specific_humidity_limiter_updates_per_level: {level_updates}")


def limit_sphum_tendency(
    sphum: xr.DataArray, dQ1: xr.DataArray, dQ2: xr.DataArray, dt: float
) -> Tuple[xr.DataArray, xr.DataArray]:
    delta = dQ2 * dt
    reduction_ratio = (-sphum) / (dt * dQ2)  # type: ignore
    dQ1_updated = xr.where(sphum + delta > 0, dQ1, reduction_ratio * dQ1)
    dQ2_updated = xr.where(sphum + delta > 0, dQ2, reduction_ratio * dQ2)
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
    ds = xr.Dataset(state)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


class MLStepper(Stepper, LoggingMixin):
    def __init__(
        self,
        fv3gfs: Any,
        comm: Any,
        timestep: float,
        states_to_output: Sequence[str],
        model: MultiModelAdapter,
        diagnostic_only: bool = False,
    ):
        self.rank: int = comm.rank
        self.comm = comm
        self._fv3gfs = fv3gfs
        self._do_only_diagnostic_ml = diagnostic_only
        self._timestep = timestep
        self._model = model
        self._states_to_output = states_to_output

        self._tendencies_to_apply_to_dycore_state: State = {}
        self._tendencies_to_apply_to_physics_state: State = {}

    def _apply_python_to_dycore_state(self) -> Diagnostics:

        updated_state: State = {}

        variables: List[Hashable] = [
            TENDENCY_TO_STATE_NAME["dQ1"],
            TENDENCY_TO_STATE_NAME["dQ2"],
            DELP,
            PRECIP_RATE,
            TOTAL_PRECIP,
        ]
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}
        tendency = self._tendencies_to_apply_to_dycore_state
        diagnostics = runtime.compute_ml_diagnostics(state, tendency)

        if self._do_only_diagnostic_ml:
            runtime.rename_diagnostics(diagnostics)
        else:
            updated_state.update(apply(state, tendency, dt=self._timestep))

        updated_state[TOTAL_PRECIP] = precipitation_sum(
            state[TOTAL_PRECIP], diagnostics["net_moistening"], self._timestep
        )

        self._log_debug("Setting Fortran State")
        self._state.update(updated_state)

        diagnostics.update({name: self._state[name] for name in self._states_to_output})

        return {
            "area": self._state[AREA],
            "cnvprcp_after_python": self._fv3gfs.get_diagnostic_by_name(
                "cnvprcp"
            ).data_array,
            "total_precip": updated_state[TOTAL_PRECIP],
            **diagnostics,
        }

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
        variables: List[Hashable] = list(set(self._model.input_variables) | {SPHUM})
        self._log_debug(f"Getting state variables: {variables}")
        state = {name: self._state[name] for name in variables}

        self._log_debug("Computing ML-predicted tendencies")
        tendency: State = predict(self._model, state)

        if "dQ2" in tendency:
            self._log_debug(
                "Correcting ML tendencies that would predict negative specific humidity"
            )
            tendency_updated: State = {
                name: tendency
                for name, tendency in zip(
                    ["dQ1", "dQ2"],
                    limit_sphum_tendency(
                        state[SPHUM],
                        tendency["dQ1"],
                        tendency["dQ2"],
                        dt=self._timestep,
                    ),
                )
            }
            log_updated_tendencies(self.comm, tendency, tendency_updated)
        else:
            tendency_updated = tendency

        self._tendencies_to_apply_to_dycore_state = {
            k: v for k, v in tendency_updated.items() if k in ["dQ1", "dQ2"]
        }
        self._tendencies_to_apply_to_physics_state = {
            k: v for k, v in tendency_updated.items() if k in ["dQu", "dQv"]
        }
        return {}
