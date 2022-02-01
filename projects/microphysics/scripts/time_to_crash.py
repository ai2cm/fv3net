import sys
import vcm
import re
import json
import datetime


def open_segmented_logs(url: str):
    fs = vcm.get_fs(url)
    logfiles = sorted(fs.glob(f"{url}/**/logs.txt"))
    logs = [fs.cat(url).decode() for url in logfiles]
    return logs


log = open_segmented_logs(sys.argv[1])[0]
profile_info = re.findall(r"INFO:profiles:(.*)", log)
profile_json = [json.loads(s) for s in profile_info]
profile_time = sorted(
    set([datetime.datetime.fromisoformat(data["time"]) for data in profile_json])
)
last_time = profile_time[-1]
initial_time = profile_time[0] - (profile_time[1] - profile_time[0])
duration = last_time - initial_time

print("Initial time:", initial_time.isoformat())
print("Final time:", last_time.isoformat())
print("Duration:", str(duration))
