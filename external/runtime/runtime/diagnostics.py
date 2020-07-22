from typing import Any, Sequence
from datetime import datetime, timedelta
from toolz import dissoc


# TODO rename and perhaps simplify this object hierarchy
class Containable:
    """base class for time selection strategies"""

    def __contains__(self, value: Any):
        return True


class SelectedTimes(Containable):
    def __init__(self, d):
        self._d = d

    @property
    def times(self) -> Sequence[str]:
        return self._d["times"]

    def __contains__(self, time: datetime):
        time_stamp = time.strftime(f"%Y%m%d.%h%m%s")
        return time_stamp in self.times


class RegularTimes(Containable):
    def __init__(self, d):
        self._d = d

    @property
    def frequency(self) -> timedelta:
        return timedelta(seconds=self._d["frequency"])

    def __contains__(self, time: datetime):
        midnight = time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = time - midnight
        return time_since_midnight % self.frequency == 0


def get_time(d):
    kind = d.get("kind", "every")
    if kind == "regular":
        return RegularTimes(d)
    elif kind == "selected":
        return SelectedTimes(d)
    else:
        return Containable()


class DiagnosticFile:
    def __init__(self, d):
        self.d = d

    @property
    def name(self):
        return self.d["name"]

    @property
    def variables(self):
        return self.d.get("variables", Containable())

    @property
    def times(self):
        return get_time(self.d.get("times", {}))


class DiagnosticConfig:
    def __init__(self, d):
        self._d = d

    @property
    def diagnostics(self):
        if len(self._d) == 0:
            return DiagnosticFile({"name": "diags.zarr"})
        else:
            return [DiagnosticFile(item) for item in self._d]

