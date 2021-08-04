from dataclasses import dataclass
import logging
from typing import (
    Callable,
    Iterable,
    Set,
)
import vcm
from runtime.types import Diagnostics, State

from .names import DELP

logger = logging.getLogger(__name__)


@dataclass
class Monitor:
    """Utility class for monitoring changes to a state dictionary and returning
    the outputs as tendencies
    """

    tendency_variables: Set[str]
    storage_variables: Set[str]
    _state: State
    timestep: float

    def __call__(
        self, name: str, func: Callable[[], Diagnostics],
    ) -> Callable[[], Diagnostics]:
        """Decorator to add tendency monitoring to an update function

        This will add the following diagnostics:
        - `tendency_of_{variable}_due_to_{name}`
        - `storage_of_{variable}_path_due_to_{name}`. A mass-integrated version
        of the above
        - `storage_of_mass_due_to_{name}`, the total mass tendency in Pa/s.

        Args:
            name: the name to tag the tendency diagnostics with
            func: a stepping function which modifies the `state` dictionary this object
                is monitoring, but does not directly modify the `DataArray` objects
                it contains
        Returns:
            monitored function. Same as func, but with tendency and mass change
            diagnostics inserted in place
        """

        def step() -> Diagnostics:

            vars_ = self.storage_variables | self.tendency_variables
            delp_before = self._state[DELP]
            before = {key: self._state[key] for key in vars_}
            diags = func()
            delp_after = self._state[DELP]
            after = {key: self._state[key] for key in vars_}

            for variable in self.storage_variables:
                path_before = vcm.mass_integrate(before[variable], delp_before, "z")
                path_after = vcm.mass_integrate(after[variable], delp_after, "z")

                diag_name = f"storage_of_{variable}_path_due_to_{name}"
                diags[diag_name] = (path_after - path_before) / self.timestep
                if "units" in before[variable].attrs:
                    diags[diag_name].attrs["units"] = (
                        before[variable].units + " kg/m**2/s"
                    )

            # Compute statistics
            for variable in self.tendency_variables:
                diag_name = f"tendency_of_{variable}_due_to_{name}"
                diags[diag_name] = (after[variable] - before[variable]) / self.timestep
                if "units" in before[variable].attrs:
                    diags[diag_name].attrs["units"] = before[variable].units + "/s"

            mass_change = (delp_after - delp_before).sum("z") / self.timestep
            mass_change.attrs["units"] = "Pa/s"
            diags[f"storage_of_mass_due_to_{name}"] = mass_change
            return diags

        # ensure monitored function has same name as original
        step.__name__ = func.__name__
        return step

    @staticmethod
    def from_variables(
        variables: Iterable[str], state: State, timestep: float
    ) -> "Monitor":
        """

        Args:
            variables: list of variables with names like
                `tendency_of_{variable}_due_to_{name}`. Used to infer the variables
                to be monitored.
            state: The mutable object to monitor
            timestep: the length of the timestep used to compute the tendency
        """
        # need to consume variables into set to use more than once
        var_set = set(variables)
        return Monitor(
            tendency_variables=filter_tendency(var_set),
            storage_variables=filter_storage(var_set),
            _state=state,
            timestep=timestep,
        )


def filter_matching(variables: Iterable[str], split: str, prefix: str) -> Set[str]:
    """Get sequences of tendency and storage variables from diagnostics config."""
    return {
        variable.split(split)[0][len(prefix) :]
        for variable in variables
        if variable.startswith(prefix) and split in variable
    }


def filter_storage(variables: Iterable[str]) -> Set[str]:
    return filter_matching(variables, split="_path_due_to_", prefix="storage_of_")


def filter_tendency(variables: Iterable[str]) -> Set[str]:
    return filter_matching(variables, split="_due_to_", prefix="tendency_of_")
