import logging
import os
import submit_utils

logger = logging.getLogger("run_jobs")


PARENT_BUCKET = (
    "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/"
)

INPUT_BUCKET = PARENT_BUCKET + "restart/C48/"
OUTPUT_BUCKET = PARENT_BUCKET + "one_step_output/C48/"
CONFIG_BUCKET = PARENT_BUCKET + "one_step_config/C48/"

PWD = os.path.dirname(os.path.abspath(__file__))
DOCKER_IMAGE = "us.gcr.io/vcm-ml/fv3gfs-python"
LOCAL_RUNFILE = os.path.join(PWD, "runfile.py")
LOCAL_DIAG_TABLE = os.path.join(PWD, "diag_table")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    timestep_list = submit_utils.timesteps_to_process(INPUT_BUCKET, OUTPUT_BUCKET)
    submit_utils.submit_jobs(
        timestep_list[0:1], CONFIG_BUCKET, LOCAL_RUNFILE,
        LOCAL_DIAG_TABLE, DOCKER_IMAGE, OUTPUT_BUCKET
    )
