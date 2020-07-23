from typing import Any, Sequence
from datetime import datetime, timedelta
import random


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
        time_stamp = time.strftime(r"%Y%m%d.%H%M%S")
        print(time_stamp)
        return time_stamp in self.times


class RegularTimes(Containable):
    def __init__(self, d):
        self._d = d

        if self.frequency > timedelta(days=1.0):
            raise ValueError("Minimum output frequency is daily.")

    @property
    def frequency(self) -> timedelta:
        return timedelta(seconds=self._d["frequency"])

    def __contains__(self, time: datetime):
        midnight = time.replace(hour=0, minute=0, second=0, microsecond=0)
        time_since_midnight = time - midnight
        quotient = time_since_midnight % self.frequency
        return quotient == timedelta(seconds=0)


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
    def diagnostics(self) -> Sequence[DiagnosticFile]:
        diags_configs = self._d.get("diagnostics", [])
        if len(diags_configs) > 0:
            return [DiagnosticFile(item) for item in diags_configs]
        else:
            # Keep old behavior for backwards compatiblity
            output_name = self._d["scikit_learn"]["zarr_output"]
            return [DiagnosticFile({"name": output_name})]
