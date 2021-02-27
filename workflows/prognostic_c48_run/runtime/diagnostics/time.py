import dataclasses
from typing import (
    Any,
    Sequence,
    Container,
    Mapping,
    List,
    Union,
    Dict,
    Optional,
    MutableMapping,
)

import cftime
import datetime


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
    includes_lower: bool = False

    def _is_endpoint(self, time: cftime.DatetimeJulian) -> bool:
        remainder = (time - self.initial_time) % self.frequency
        return remainder == datetime.timedelta(0)

    def indicator(self, time: cftime.DatetimeJulian) -> Optional[cftime.DatetimeJulian]:
        n = (time - self.initial_time) // self.frequency

        if self._is_endpoint(time) and not self.includes_lower:
            n = n - 1

        return n * self.frequency + self.frequency / 2 + self.initial_time


@dataclasses.dataclass
class TimeConfig:
    """Configuration for output times

    This class configures the time coordinate of the output diagnostics. It
    allows output data at a user-specified list of snapshots
    (``kind='selected'``), fixed intervals (``kind='interval'``), averages
    over intervals (``kind='interval-average'``), or every single time step
    (``kind='every'``).

    Attributes:
        kind: one of interval, every, "interval-average", or "selected"
        times: List of times to be used when kind=="selected". The times
            should be formatted as YYYYMMDD.HHMMSS strings. Example:
            ``["20160101.000000"]``.
        frequency: frequency in seconds, used for kind=interval-average or interval
        includes_lower: for interval-average, whether the interval includes its upper
            or lower limit. Default: False.
    """

    frequency: Optional[float] = None
    times: Optional[List[str]] = None
    kind: str = "every"
    includes_lower: bool = False

    def time_container(self, initial_time: cftime.DatetimeJulian) -> TimeContainer:
        if self.kind == "interval" and self.frequency:
            return TimeContainer(IntervalTimes(self.frequency, initial_time))
        elif self.kind == "selected":
            return TimeContainer(SelectedTimes(self.times or []))
        elif self.kind == "every":
            return TimeContainer(All())
        elif self.kind == "interval-average" and self.frequency:
            return IntervalAveragedTimes(
                datetime.timedelta(seconds=self.frequency),
                initial_time,
                self.includes_lower,
            )
        else:
            raise NotImplementedError(f"Time {self.kind} not implemented.")
