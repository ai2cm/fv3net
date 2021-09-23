import dataclasses
import logging
from typing import Hashable, Mapping, MutableMapping, Set

import xarray as xr

import vcm
import fv3fit
import fv3gfs.util
from runtime.monitor import Monitor
from runtime.types import Diagnostics, Step, State
from runtime.steppers.machine_learning import predict
from runtime.names import DELP, TEMP, SPHUM

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MLPrescriberConfig:
    """Configuration for overriding tendencies from a step with ML predictions.
    
    Attributes:
        url: path to dataset containing tendencies.
        variables: mapping from state name to name of corresponding tendency in
            provided dataset. For example: {"air_temperature": "fine_res_Q1"}.
    """

    url: str
    variables: Mapping[str, str]


@dataclasses.dataclass
class MLPrescriberAdapter:
    """Wrap a Step function and prescribe certain tendencies."""

    config: MLPrescriberConfig
    state: State
    timestep: float
    diagnostic_variables: Set[str] = dataclasses.field(default_factory=set)

    def __post_init__(self: "MLPrescriberAdapter"):
        logger.debug(f"Opening ML model for overriding from: {self.config.url}")
        self._model = fv3fit.load(self.config.url)

    @property
    def monitor(self) -> Monitor:
        return Monitor.from_variables(
            self.diagnostic_variables, self.state, self.timestep
        )

    def _prescribe_tendency(self, name: str, func: Step) -> Diagnostics:
        tendencies = predict(self._model, self.state)
        before = self.monitor.checkpoint()
        diags = func()
        change_due_to_func = self.monitor.compute_change(name, before, self.state)
        logger.debug(f"Overriding tendencies from {name} with ML predictions.")
        for variable_name, tendency_name in self.config.variables.items():
            with xr.set_options(keep_attrs=True):
                self.state[variable_name] = (
                    before[variable_name] + tendencies[tendency_name] * self.timestep
                )
        change_due_to_ml = self.monitor.compute_change(
            "machine_learning", before, self.state
        )
        states_as_diags = {k: self.state[k] for k in [TEMP, SPHUM, DELP]}
        diags["net_moistening_due_to_machine_learning"] = vcm.mass_integrate(
            change_due_to_ml["tendency_of_specific_humidity_due_to_machine_learning"],
            self.state[DELP],
            dim="z",
        ).assign_attrs(units="kg/m^2/s")
        heating = vcm.column_integrated_heating_from_isochoric_transition(
            change_due_to_ml["tendency_of_air_temperature_due_to_machine_learning"],
            self.state[DELP],
            "z",
        ).assign_attrs(units="W/m^2")
        diags["column_heating_due_to_machine_learning"] = heating
        return {**diags, **change_due_to_func, **change_due_to_ml, **states_as_diags}

    def __call__(self, name: str, func: Step) -> Step:
        """Override tendencies from a function that updates the State.
        
        Args:
            name: a label for the step that is being overidden.
            func: a function that updates the State and return Diagnostics.
            
        Returns:
            overridden_func: a function which observes the change to State
            done by ``func`` and prescribes the change for specified variables.
        """

        def step() -> Diagnostics:
            return self._prescribe_tendency(name, func)

        step.__name__ = func.__name__
        return step
