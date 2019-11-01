from typing import Set
import subprocess
from os.path import basename
import logging

logging.basicConfig(level=logging.INFO)

INPUT_BUCKET = 'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/coarsened/C48'
OUTPUT_BUCKET = 'gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/restart/C48'

def timestep(path):
    if path.endswith('/'):
        return timestep(path[:-1])
    else:
        return basename(path)


def list_time_steps(bucket: str) -> Set:
    bytes = subprocess.check_output(['gsutil', 'ls', bucket])
    string = bytes.decode('UTF-8')
    files = string.strip().split()
    return set(map(timestep, files))
    

def timesteps_to_process() -> Set:
    to_do = list_time_steps(INPUT_BUCKET)
    done = list_time_steps(OUTPUT_BUCKET)
    logging.info('Number of input times: %s' % len(to_do))
    logger.info(f'Number of completed times: {len(done)}')
    logging.info('Number of times to process: %d' %  (len(to_do) - len(done)))

    return to_do - done


for line in timesteps_to_process():
    print(line)
