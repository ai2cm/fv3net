from dataclasses import dataclass
import logging
from typing import (
    Callable,
    Iterable,
    Mapping,
    Hashable,
    Set,
)
import xarray as xr
from runtime.types import Diagnostics, State
import vcm
from runtime.names import DELP

logger = logging.getLogger(__name__)

Checkpoint = Mapping[Hashable, xr.DataArray]


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

        This will add the following diagnostics and state variables:
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
            before = self.checkpoint()
            diags = func()
            after = self.checkpoint()
            changes = self.compute_change(name, before, after)
            for key in changes:
                self._state[key] = changes[key]
            diags.update(changes)
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

    def checkpoint(self) -> Checkpoint:
        """Copy the monitored variables into a new dictionary """
        vars_ = list(
            set(self.tendency_variables) | set(self.storage_variables) | {DELP}
        )
        return {key: self._state[key] for key in vars_}

    def compute_change(
        self, name: str, before: Checkpoint, after: Checkpoint
    ) -> Diagnostics:
        """Compute the change between two checkpoints

        Args:
            name: labels the output variable names. Same meaning as in __call__
            before: the initial state
            after: the final state

        Returns:
            storage and tendencies computed between before and after

        Examples:
            >>> before = monitor.checkpoint()
            >>> # some changes
            >>> after = monitor.checkpoint()
            >>> storage_and_tendencies = monitor.compute_change("label", before, after)

        """
        return compute_change(
            before,
            after,
            self.tendency_variables,
            self.storage_variables,
            name,
            self.timestep,
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


def compute_change(
    before: Checkpoint,
    after: Checkpoint,
    tendency_variables: Set[str],
    storage_variables: Set[str],
    name: str,
    timestep: float,
):
    diags = {}
    delp_before = before[DELP]
    delp_after = after[DELP]
    # Compute statistics
    for variable in tendency_variables:
        diag_name = f"tendency_of_{variable}_due_to_{name}"
        diags[diag_name] = (after[variable] - before[variable]) / timestep
        if "units" in before[variable].attrs:
            diags[diag_name].attrs["units"] = before[variable].units + "/s"

    for variable in storage_variables:
        path_before = vcm.mass_integrate(before[variable], delp_before, "z")
        path_after = vcm.mass_integrate(after[variable], delp_after, "z")

        diag_name = f"storage_of_{variable}_path_due_to_{name}"
        diags[diag_name] = (path_after - path_before) / timestep
        if "units" in before[variable].attrs:
            diags[diag_name].attrs["units"] = before[variable].units + " kg/m**2/s"

    mass_change = (delp_after - delp_before).sum("z") / timestep
    mass_change.attrs["units"] = "Pa/s"
    diags[f"storage_of_mass_due_to_{name}"] = mass_change
    return diags
