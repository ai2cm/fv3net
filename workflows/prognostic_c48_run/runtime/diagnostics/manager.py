from typing import Any, Sequence, Container, Mapping, List, Union, Dict, Optional

import datetime
import cftime
import logging
import fv3gfs.util
import xarray as xr
import dataclasses

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


class TimeContainer:
    """A time discretization can be described by an "indicator" function
    mapping times onto discrete set of output times.

    This generalizes the notion of a set of times to include a concept of grouping.

    """

    def __init__(self, container: Container):
        self.container = container

    def indicator(self, time: cftime.DatetimeJulian) -> Optional[cftime.DatetimeJulian]:
        """Maps a value onto set"""
        if time in self.container:
            return time
        else:
            return None


@dataclasses.dataclass
class IntervalAveragedTimes(TimeContainer):
    frequency: datetime.timedelta
    initial_time: cftime.DatetimeJulian

    def indicator(self, time: cftime.DatetimeJulian) -> Optional[cftime.DatetimeJulian]:
        n = (time - self.initial_time) // self.frequency
        return n * self.frequency + self.frequency / 2 + self.initial_time


@dataclasses.dataclass
class DiagnosticFileConfig:
    """
    Attrs:
        name: file name of a zarr to store the data in, e.g., 'diags.zarr'
        variables: a container of variables to save
        times (optional): a container for times to output
    """

    name: str
    variables: Container
    times: TimeContainer = TimeContainer(All())

    @classmethod
    def from_dict(
        cls, dict_: Mapping, initial_time: cftime.DatetimeJulian
    ) -> "DiagnosticFileConfig":
        return DiagnosticFileConfig(
            name=dict_["name"],
            variables=dict_.get("variables", All()),
            times=cls._get_times(dict_.get("times", {}), initial_time),
        )

    def to_dict(self) -> Dict:
        return {"name": self.name, "variables": self.variables, "times": self.times}

    @staticmethod
    def _get_times(d, initial_time: cftime.DatetimeJulian) -> TimeContainer:
        kind = d.get("kind", "every")
        if kind == "interval":
            return TimeContainer(IntervalTimes(d["frequency"], initial_time))
        elif kind == "selected":
            return TimeContainer(SelectedTimes(d["times"]))
        elif kind == "every":
            return TimeContainer(All())
        elif kind == "interval-average":
            return IntervalAveragedTimes(
                datetime.timedelta(seconds=d["frequency"]), initial_time=initial_time,
            )
        else:
            raise NotImplementedError(f"Time {kind} not implemented.")


class DiagnosticFile:
    """A object representing a time averaged diagnostics file

    Provides a similar interface as the "diag_table"

    Replicates the abilities of the fortran models's diag_table by allowing
    the user to specify different output times for distinct sets of
    variables.

    Note:
        Outputing a snapshot is type of time-average (e.g. taking the average
        with respect to a point mass at a given time).

    """

    def __init__(
        self,
        variables: Container,
        monitor: fv3gfs.util.ZarrMonitor,
        times: TimeContainer,
    ):
        """

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
        self.variables = variables
        self.times = times
        self._monitor = monitor

        # variables used for averaging
        self._running_total: Dict[str, xr.DataArray] = {}
        self._current_label: Optional[cftime.DatetimeJulian] = None
        self._n = 0
        self._units: Dict[str, str] = {}

    def observe(
        self, time: cftime.DatetimeJulian, diagnostics: Mapping[str, xr.DataArray]
    ):
        for key in diagnostics:
            self._units[key] = diagnostics[key].attrs.get("units", "unknown")

        label = self.times.indicator(time)
        if label is not None:
            if label != self._current_label:
                self.flush()
                self._reset_running_average(label, diagnostics)
            else:
                self._increment_running_average(diagnostics)

    def _reset_running_average(self, label, diagnostics):
        self._running_total = {key: val.copy() for key, val in diagnostics.items()}
        self._current_label = label
        self._n = 1

    def _increment_running_average(self, diagnostics):
        for key in diagnostics:
            self._running_total[key] += diagnostics[key]
            self._n += 1

    def flush(self):
        if self._current_label is not None:
            average = {key: val / self._n for key, val in self._running_total.items()}
            quantities = {
                # need units for from_data_array to work
                key: fv3gfs.util.Quantity.from_data_array(
                    average[key].assign_attrs(units=self._units[key])
                )
                for key in average
                if key in self.variables
            }

            # patch this in manually. the ZarrMonitor needs it.
            # We should probably modify this behavior.
            quantities["time"] = self._current_label
            self._monitor.store(quantities)

    def __del__(self):
        self.flush()


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
    diag_dicts = config.get("diagnostics", [])
    configs: List[DiagnosticFileConfig] = []

    if len(diag_dicts) > 0:
        for diag_dict in diag_dicts:
            configs.append(
                DiagnosticFileConfig.from_dict(diag_dict, initial_time=initial_time)
            )
    else:
        default_config = DiagnosticFileConfig(
            name=config["scikit_learn"]["zarr_output"],
            times=TimeContainer(All()),
            variables=All(),
        )
        configs.append(default_config)

    return [
        DiagnosticFile(
            variables=config.variables,
            times=config.times,
            monitor=fv3gfs.util.ZarrMonitor(config.name, partitioner, mpi_comm=comm),
        )
        for config in configs
    ]
