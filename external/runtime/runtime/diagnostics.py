from typing import Any, Sequence, Container, Mapping, List, Union
from datetime import datetime, timedelta
import fv3util


class All(Container):
    """A container that contains every thing
    
    This is useful for cases where we want an ``in`` check to always return True.

    Example:
        >>> all = All()
        >>> 'x' in all
        True
        >>> 1232.1 in all
        True
    """

    def __contains__(self, value: Any) -> bool:
        return True


class SelectedTimes(Container[datetime]):
    TIME_FMT: str = r"%Y%m%d.%H%M%S"

    def __init__(self, times=Sequence[str]):
        self._time_stamps = times

        # see if there is an error
        self.times

    @property
    def times(self) -> Sequence[datetime]:
        return [datetime.strptime(time, self.TIME_FMT) for time in self._time_stamps]

    def __contains__(self, time) -> bool:
        return time in self.times


class RegularTimes(Container[datetime]):
    def __init__(self, frequency_seconds: Union[float, int]):
        self._frequency_seconds = frequency_seconds
        if self.frequency > timedelta(days=1.0):
            raise ValueError("Minimum output frequency is daily.")

    @property
    def frequency(self) -> timedelta:
        return timedelta(seconds=self._frequency_seconds)

    def __contains__(self, time) -> bool:
        midnight = time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = time - midnight
        quotient = time_since_midnight % self.frequency
        return quotient == timedelta(seconds=0)


class DiagnosticFile:
    """A object representing a diagnostics file

    Provides a similar interface as the "diag_table"
    """

    def __init__(
        self,
        name: str,
        times: Container[datetime],
        variables: Container,
        partitioner,
        comm,
    ):
        self.name = name
        self.times = times
        self.variables = variables
        self._monitor = fv3util.ZarrMonitor(self.name, partitioner, mpi_comm=comm)

    def observe(self, time: datetime, diagnostics: Mapping):
        quantities = {
            # need units for from_data_array to work
            key: fv3util.Quantity.from_data_array(
                diagnostics[key].assign_attrs(units="unknown")
            )
            for key in diagnostics
            if key in self.variables
            if time in self.times
        }

        # patch this in manually. the ZarrMonitor needs it.
        # We should probably modify this behavior.
        quantities["time"] = time
        self._monitor.store(quantities)


def _get_times(d) -> Container[datetime]:
    kind = d.get("kind", "every")
    if kind == "regular":
        return RegularTimes(d["frequency"])
    elif kind == "selected":
        return SelectedTimes(d["times"])
    elif kind == "every":
        return All()
    else:
        raise NotImplementedError(f"Time {kind} not implemented.")


def _config_to_diagnostic_file(
    diag_file_config: Mapping, partitioner, comm
) -> DiagnosticFile:
    return DiagnosticFile(
        name=diag_file_config["name"],
        variables=diag_file_config.get("variables", All()),
        times=_get_times(diag_file_config.get("times", {})),
        partitioner=partitioner,
        comm=comm,
    )


def get_diagnostic_files(
    config, partitioner: fv3util.CubedSpherePartitioner, comm
) -> List[DiagnosticFile]:
    """Initialize a list of diagnostic file objects from a configuration dictionary

    Args:
        config: A loaded "fv3config" object with a "diagnostics" section
        paritioner: a partioner object used for writing, maybe it would be
            cleaner to pass a factory
        comm: an MPI Comm object

    """
    diag_configs = config.get("diagnostics", [])
    if len(diag_configs) > 0:
        return [
            _config_to_diagnostic_file(config, partitioner, comm)
            for config in diag_configs
        ]
    else:
        # Keep old behavior for backwards compatiblity
        output_name = config["scikit_learn"]["zarr_output"]
        return [
            DiagnosticFile(
                name=output_name,
                variables=All(),
                times=All(),
                partitioner=partitioner,
                comm=comm,
            )
        ]
