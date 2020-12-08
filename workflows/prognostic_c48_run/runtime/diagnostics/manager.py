from typing import Any, Sequence, Container, Mapping, List, Union, Optional, Dict

import datetime
import cftime
import logging
import fv3gfs.util
import xarray as xr

logger = logging.getLogger(__name__)


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


class SelectedTimes(Container[cftime.DatetimeJulian]):
    TIME_FMT: str = r"%Y%m%d.%H%M%S"

    def __init__(self, times=Sequence[str]):
        self._time_stamps = times

        # see if there is an error
        self.times

    @property
    def _times(self) -> Sequence[datetime.datetime]:
        return [
            datetime.datetime.strptime(time, self.TIME_FMT)
            for time in self._time_stamps
        ]

    @property
    def times(self) -> Sequence[cftime.DatetimeJulian]:
        return [cftime.DatetimeJulian(*time.timetuple()) for time in self._times]

    def __contains__(self, time: cftime.DatetimeJulian) -> bool:
        return time in self.times


class IntervalTimes(Container[cftime.DatetimeJulian]):
    def __init__(
        self, frequency_seconds: Union[float, int], initial_time: cftime.DatetimeJulian,
    ):
        """
        Args:
            frequency_seconds: the output frequency from the initial time
            initial_time: the initial time to start the period
            
        """
        self._frequency_seconds = frequency_seconds
        self.initial_time = initial_time
        if self.frequency > datetime.timedelta(days=1.0) and initial_time is None:
            raise ValueError(
                "Minimum output frequency is daily when intial_time is not provided."
            )

    @property
    def frequency(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=self._frequency_seconds)

    def __contains__(self, time) -> bool:
        time_since_initial_time = time - self.initial_time
        quotient = time_since_initial_time % self.frequency
        return quotient == datetime.timedelta(seconds=0)


def _assign_units_if_none_present(array: xr.DataArray, units=None):
    return array.assign_attrs(units=array.attrs.get("units", units))


class DiagnosticFile:
    """A object representing a diagnostics file

    Provides a similar interface as the "diag_table"

    Replicates the abilities of the fortran models's diag_table by allowing
    the user to specify different output times for distinct sets of
    variables.
    """

    def __init__(
        self, name: str, times: Container[cftime.DatetimeJulian], variables: Container,
    ):
        """
        Args:
            monitor: an underlying monitor to store the data in
            times: the set of times (potentially infinite) to save the data at
            variables: a container of variables to save

        Note:

            The containers used for times and variables do not need to be
            concrete lists or python sequences. They only need to satisfy the
            abstract ``Container`` interface. Please see the special
            containers for outputing times above:

            - ``IntervalTimes``
            - ``SelectedTimes``

            as well as the generic ``All`` container that contains the entire
            Universe!
        """
        self._name = name
        self._monitor: Optional[fv3gfs.util.ZarrMonitor] = None
        self.times = times
        self.variables = variables

    def observe(
        self, time: cftime.DatetimeJulian, diagnostics: Mapping[str, xr.DataArray]
    ):
        """Possibly store the data into the monitor
        """
        if self._monitor is None:
            raise ValueError(
                f"zarr monitor not yet established for {self._name}. Call set_monitor."
            )
        if time in self.times:
            quantities = {
                # need units for from_data_array to work
                key: fv3gfs.util.Quantity.from_data_array(
                    _assign_units_if_none_present(diagnostics[key], "unknown")
                )
                for key in diagnostics
                if key in self.variables
            }

            # patch this in manually. the ZarrMonitor needs it.
            # We should probably modify this behavior.
            quantities["time"] = time
            self._monitor.store(quantities)

    @classmethod
    def from_config(
        cls, diag_file_config: Mapping, initial_time: cftime.DatetimeJulian
    ) -> "DiagnosticFile":
        return DiagnosticFile(
            name=diag_file_config["name"],
            variables=diag_file_config.get("variables", All()),
            times=cls._get_times(diag_file_config.get("times", {}), initial_time),
        )

    def set_monitor(
        self, partitioner: fv3gfs.util.CubedSpherePartitioner, comm
    ) -> "DiagnosticFile":
        if self._monitor is not None:
            raise ValueError(f"zarr monitor already initialized at {self._name}")
        self._monitor = fv3gfs.util.ZarrMonitor(self._name, partitioner, mpi_comm=comm)
        return self

    @staticmethod
    def _get_times(
        d, initial_time: cftime.DatetimeJulian
    ) -> Container[cftime.DatetimeJulian]:
        kind = d.get("kind", "every")
        if kind == "interval":
            return IntervalTimes(d["frequency"], initial_time)
        elif kind == "selected":
            return SelectedTimes(d["times"])
        elif kind == "every":
            return All()
        else:
            raise NotImplementedError(f"Time {kind} not implemented.")

    def to_dict(self) -> Dict:
        return {"name": self._name, "variables": self.variables, "times": self.times}


def get_diagnostic_files(
    config: Mapping,
    partitioner: fv3gfs.util.CubedSpherePartitioner,
    comm,
    initial_time: cftime.DatetimeJulian,
) -> List[DiagnosticFile]:
    """Initialize a list of diagnostic file objects from a configuration dictionary
    Note- the default here is to save all the variables in the diagnostics.
    The default set of variables can be overwritten by inserting a default diagnostics
    config entry for each runfile, e.g. ../prepare_config.py does this for
    the sklearn runfile.

    Args:
        config: A loaded "fv3config" dictionary with a "diagnostics" section
        paritioner: a partioner object used for writing, maybe it would be
            cleaner to pass a factory
        comm: an MPI Comm object
        initial_time: the initial time of the simulation.

    """
    diag_configs = config.get("diagnostics", [])
    if len(diag_configs) > 0:
        return [
            DiagnosticFile.from_config(config, initial_time).set_monitor(
                partitioner, comm
            )
            for config in diag_configs
        ]
    else:
        # Keep old behavior for backwards compatiblity
        output_name = config["scikit_learn"]["zarr_output"]
        default_config = {"name": output_name, "times": {}, "variables": All()}
        return [
            DiagnosticFile.from_config(default_config, initial_time).set_monitor(
                partitioner, comm
            )
        ]
