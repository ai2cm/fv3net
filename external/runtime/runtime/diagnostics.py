from typing import Any, Sequence, Container, Mapping, List
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

    def __init__(self, d):
        self._d = d

        # see if there are any errors
        self.times

    @property
    def times(self) -> Sequence[datetime]:
        return [datetime.strptime(time, self.TIME_FMT) for time in self._d["times"]]

    def __contains__(self, time) -> bool:
        return time in self.times


class RegularTimes(Container[datetime]):
    def __init__(self, d):
        self._d = d

        if self.frequency > timedelta(days=1.0):
            raise ValueError("Minimum output frequency is daily.")

    @property
    def frequency(self) -> timedelta:
        return timedelta(seconds=self._d["frequency"])

    def __contains__(self, time) -> bool:
        midnight = time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = time - midnight
        quotient = time_since_midnight % self.frequency
        return quotient == timedelta(seconds=0)


class DiagnosticFile:
    """A object representing a diagnostics file

    Provides a similar interface as the "diag_table"
    """

    def __init__(self, d: Mapping, partitioner, comm):
        self.d = d
        self._monitor = fv3util.ZarrMonitor(self.name, partitioner, mpi_comm=comm)

    @property
    def name(self):
        return self.d["name"]

    @property
    def variables(self) -> Container:
        return self.d.get("variables", All())

    @property
    def times(self) -> Container[datetime]:
        kind = self.d.get("kind", "every")
        if kind == "regular":
            return RegularTimes(self.d)
        elif kind == "selected":
            return SelectedTimes(self.d)
        else:
            return All()

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


def get_diagnostic_files(_d, partitioner, comm) -> List[DiagnosticFile]:
    diags_configs = _d.get("diagnostics", [])
    if len(diags_configs) > 0:
        return [DiagnosticFile(item, partitioner, comm) for item in diags_configs]
    else:
        # Keep old behavior for backwards compatiblity
        output_name = _d["scikit_learn"]["zarr_output"]
        return [DiagnosticFile({"name": output_name}, partitioner, comm)]
