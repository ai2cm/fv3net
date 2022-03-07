from typing import List
import re
import json
import datetime


def parse_duration(logs: List[str]) -> datetime.timedelta:
    log = "".join(logs)
    profile_info = re.findall(r"INFO:profiles:(.*)", log)
    profile_json = [json.loads(s) for s in profile_info]
    profile_time = sorted(
        set([datetime.datetime.fromisoformat(data["time"]) for data in profile_json])
    )
    last_time = profile_time[-1]
    initial_time = profile_time[0] - (profile_time[1] - profile_time[0])
    duration = last_time - initial_time
    return duration
