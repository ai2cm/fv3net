#!/usr/bin/env python
"""
https://cloud.google.com/stackdriver/docs/solutions/gke/managing-logs

Description of special payload strings
https://cloud.google.com/logging/docs/structured-logging#special-payload-fields
"""
from dataclasses import dataclass
import re
from enum import Enum
from typing import Mapping, Optional
import json
from typing import Any, Iterable

__all__ = ["handle_fv3_log"]


class LineType(Enum):
    # need to be letters for named groups to work
    PYTHON_LOG = "python"
    MAX_MIN = "max_min"
    FV3_LOG = "fortran"


# regexp for floating point number
floating_point = r"[-+]?[0-9]*(\.[0-9]+(E-?\d+)?)?"

REGEX = {
    LineType.PYTHON_LOG: re.compile(
        r"(?P<severity>(INFO|WARNING|ERROR|DEBUG|CRITICAL))"
        ":"
        r"(?P<module>.*?)"
        ":"
        r"(?P<message>.*)$"
    ),
    LineType.MAX_MIN: re.compile(
        r"(?P<max_min_name>.*?)"
        r"\s*max\s*=\s*"
        + (r"(?P<max>" + floating_point + ")")
        + r"\s*min\s*=\s*"
        + (r"(?P<min>" + floating_point + ")")
    ),
    LineType.FV3_LOG: re.compile("(?P<message>.*$)"),
}


@dataclass
class LogLine:
    type: LineType
    data: dict
    line: str


def parse_line(line):
    for line_type in LineType:
        match = REGEX[line_type].match(line)
        if match:
            return LogLine(line_type, match.groupdict(), line)
    raise ValueError(f"Unable to parse line: {line}")


class Handler:
    """Handler handels the parsed lines produces by the log stream

    Attributes:
        model_time: the last handled time
    """

    LABEL_NAME = "logging.googleapis.com/labels"

    def __init__(self, labels=Mapping[str, Any]):
        """
        Args:
            labels: labels to make available for query in the Google Cloud Logging
                console
        """
        self.model_time: Optional[str] = None
        self.labels = labels

    def _insert_model_time(self, payload):
        payload[self.LABEL_NAME]["model_time"] = self.model_time
        payload["model_time"] = self.model_time

    def handle(self, line: LogLine):
        payload = {**line.data}
        payload[self.LABEL_NAME] = {"kind": line.type.value, **self.labels}
        if line.type == LineType.PYTHON_LOG:
            message = payload.pop("message")
            try:
                payload["json"] = json.loads(message)
            except json.JSONDecodeError:
                payload["message"] = message

        if line.type == LineType.FV3_LOG:
            payload["severity"] = "DEBUG"

        self.model_time = payload.get("json", {}).get("time") or self.model_time

        self._insert_model_time(payload)
        return json.dumps(payload)


def handle_fv3_log(f: Iterable[str], labels: Mapping[str, Any] = {},) -> Iterable[str]:
    """Consume a fv3 log stream and produce a json stream

    Args:

        f: a iterable of lines ... e.g. a file object or sys.stdin
        labels: the labels to make available for query in the Google Cloud Logging
        output_stream: where to write the transformed data to e.g. sys.stdout

    Examples:

        Here are some jq filters that can processe these outputs.

        To print the fortran log lines only::

            jq -r 'select(.["logging.googleapis.com/labels"].kind == "fortran" ) \
                | .message'

        To print just the water vapor path and time::

            jq -r 'select(.json.water_vapor_path )  \
            | {time: .model_time, wvp: .json.water_vapor_path}'

        To show all min/max variables::

            jq -sr '[.[].max_min_name | select(. != null) | . ]  |  unique'

    """  # noqa
    handler = Handler(labels)
    for line in f:
        yield handler.handle(parse_line(line))
