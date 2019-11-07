from typing import Set
from pathlib import Path
import subprocess
from os.path import basename
import logging

logger = logging.getLogger("list_timesteps_to_run")

logging.basicConfig(level=logging.INFO)

INPUT_BUCKET = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/C48"
OUTPUT_BUCKET = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/restart/C48"


def timestep_from_url(url):
    return str(Path(url).name)


def list_time_steps(bucket: str) -> Set:
    """Returns the unique timesteps in a bucket"""
    try:
        ls_output = subprocess.check_output(["gsutil", "ls", bucket])
    except subprocess.CalledProcessError:
        return set()

    file_list = ls_output.decode("UTF-8").strip().split()
    timesteps = map(timestep_from_url, file_list)
    return set(timesteps)


def timesteps_to_process() -> Set:
    to_do = list_time_steps(INPUT_BUCKET)
    done = list_time_steps(OUTPUT_BUCKET)
    logger.info(f"Number of input times: {len(to_do)}")
    logger.info(f"Number of completed times: {len(done)}")
    logger.info(f"Number of times to process: {len(to_do) - len(done)}")

    return to_do - done


for line in timesteps_to_process():
    print(line)
