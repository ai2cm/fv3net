import argparse
import glob
import datetime
import os
import pathlib
import subprocess

from typing import List

TIME_FORMAT = "%Y%m%d.%H%M%S"

def is_time(item: str) -> bool:
    try:
        val = datetime.datetime.strptime(item, TIME_FORMAT)
        return True
    except ValueError:
        return False

def get_unique_times(path) -> List[str]:
    files = glob.glob(os.path.join(path, "*.nc"))
    unique_times = set()
    for fpath in files:
        time_prefix = pathlib.Path(fpath).name[:13]
        if is_time(time_prefix):
            unique_times.update([time_prefix])
    
    return sorted(unique_times)


def push_time_to_gcs(restart_dir, times: List[str], output_url):

    for time in times:
        cmd = ["gsutil", "-m", "cp", "-r"]
        file = os.path.join(restart_dir, f"{time}*")
        output = os.path.join(output_url, time)

        cmd.extend([file, output])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("restart_dir", type=str)
    parser.add_argument("output_url", type=str)
    args = parser.parse_args()

    times = get_unique_times(args.restart_dir)

    push_time_to_gcs(args.restart_dir, times, args.output_url)
