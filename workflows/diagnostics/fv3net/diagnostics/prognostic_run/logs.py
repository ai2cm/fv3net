from typing import Iterable, List
import re
import json
import datetime


def _parse_duration_no_json(logs: List[str]) -> List[datetime.datetime]:
    log = "".join(logs)
    profile_info = re.findall(r"INFO:profiles:(.*)", log)
    profile_json = [json.loads(s) for s in profile_info]
    return [datetime.datetime.fromisoformat(data["time"]) for data in profile_json]


def _parse_duration_json_formatted(logs: List[str]) -> Iterable[datetime.datetime]:
    combined_logs = "\n".join(logs)
    for line in combined_logs.split("\n"):
        if line.strip():
            parsed_line = json.loads(line)
            model_time = parsed_line.get("model_time", None)
            if model_time:
                yield datetime.datetime.fromisoformat(model_time)


def parse_duration(logs: List[str]) -> datetime.timedelta:
    exceptions = []

    for parse in [_parse_duration_no_json, _parse_duration_json_formatted]:
        try:
            times = sorted(parse(logs))
        except Exception as e:
            exceptions.append(e)
            pass
    if times is None:
        raise ValueError(
            f"Could not parse times from logs. Raised these exceptions {exceptions}."
        )

    last_time = times[-1]
    initial_time = times[0] - (times[1] - times[0])
    duration = last_time - initial_time
    return duration
