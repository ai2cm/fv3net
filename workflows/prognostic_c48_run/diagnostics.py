from datetime import datetime, timedelta
from toolz import dissoc

class TimeConfig:
    """base class for time selection strategies"""
    pass


class SelectedTimes(TimeConfig):
    times: Sequence[str]

    def __contains__(self, time: datetime):
        time_stamp = time.strptime(f"%Y%m%d.%h%m%s")
        return time_stamp in times

class RegularTimes(TimeConfig):
    frequency: timedelta

    def __init__(self, d):
        self._d = d

    @property
    def frequency(self):
        return timedelta(seconds=self._d['frequency'])

    def __contains__(self, time: datetime):
        midnight = time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = time - midnight
        return time_since_midnight % self.frequency == 0


class EveryTime(TimeConfig):
    def __contains__(self, time):
        return True


def get_time(d):
    kind = d.get("kind", "every")
    if kind == 'regular':
        return RegularTimes(d)
    elif kind == "selected":
        return SelectedTimes(d)
    elif kind == "every":
        return EveryTime()


class DiagnosticFile:
    name: str
    variables: Sequence[str]
    times: TimeConfig

    def __init__(self, d):
        self.d = d

    @property
    def name(self):
        return self.d['name']

    @property
    def variables(self):
        return self.d['variables']

    @property
    def times(self):
        return get_time(self.d['time'])


class DiagnosticConfig:
    diagnostics: Sequence[Diag]

    def __init__(self, d):
        self._d = d

    @property
    def diagnostics(self):
        return [
            DiagnosticFile(item)
            for item in self._d
        ]