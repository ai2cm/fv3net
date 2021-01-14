from typing import Any, Hashable, List, cast
import copy
import logging

import fv3fit
import runtime
import xarray as xr

from runtime.steppers.base import Stepper, State, Diagnostics, apply, precipitation_sum

from runtime.names import (
    SPHUM,
    DELP,
    TOTAL_PRECIP,
    PRECIP_RATE,
    AREA,
    TENDENCY_TO_STATE_NAME,
)

logger = logging.getLogger(__name__)


class MLStepper(Stepper):
    def __init__(
        self,
        fv3gfs: Any,
        comm: Any,
        timestep: float,
        states_to_output: Any,
        model: fv3fit.Predictor,
        diagnostic_only: bool = False,
    ):
        self.rank: int = comm.rank
        self.comm = comm
        self._fv3gfs: Any = fv3gfs
        self._do_only_diagnostic_ml: bool = diagnostic_only
        self._timestep: float = timestep
        self._model: fv3fit.Predictor = model
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
        tendency = predict(self._model, state)

        self._log_debug(
            "Correcting ML tendencies that would predict negative specific humidity"
        )
        tendency_updated = limit_sphum_tendency(state, tendency, dt=self._timestep)
        log_updated_tendencies(self.comm, tendency, tendency_updated)

        self._tendencies_to_apply_to_dycore_state = {
            k: v for k, v in tendency_updated.items() if k in ["dQ1", "dQ2"]
        }
        self._tendencies_to_apply_to_physics_state = {
            k: v for k, v in tendency_updated.items() if k in ["dQu", "dQv"]
        }
        return {}


def log_updated_tendencies(comm, tendency: State, tendency_updated: State):
    rank_updated_points = xr.where(tendency["dQ2"] != tendency_updated["dQ2"], 1, 0)
    updated_points = comm.reduce(rank_updated_points, root=0)
    if comm.rank == 0:
        level_updates = {
            i: int(value)
            for i, value in enumerate(updated_points.sum(["x", "y"]).values)
        }
        logger.info(f"specific_humidity_limiter_updates_per_level: {level_updates}")


def predict(model: fv3fit.Predictor, state: State) -> State:
    """Given ML model and state, return tendency prediction."""
    ds = xr.Dataset(state)  # type: ignore
    output = model.predict_columnwise(ds, feature_dim="z")
    return {key: cast(xr.DataArray, output[key]) for key in output.data_vars}


def limit_sphum_tendency(state: State, tendency: State, dt: float):
    delta = tendency["dQ2"] * dt
    tendency_updated = copy.copy(tendency)
    tendency_updated["dQ2"] = xr.where(
        state[SPHUM] + delta > 0, tendency["dQ2"], -state[SPHUM] / dt,  # type: ignore
    )
    return tendency_updated
