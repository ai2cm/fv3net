import logging
import tempfile
import os

from vcm.cloud import gsutil
import utils

"""
Script downloads files from a C48 resolution restart directory and
creates a checksum of the files.  The checksum is then uploaded to
GCS in a file for quick access.  Regression test for the coarsening,
should be updated when our coarsening technique is updated.
"""

logger = logging.getLogger(__name__)


def upload_timestep_files_and_checksum(
    source_data: str, timestep: str, checksum_folder_destination: str,
):
    with tempfile.TemporaryDirectory() as tmpdir:

        timestep_path = os.path.join(source_data, timestep)
        logger.info(f"Source timestep: {timestep_path}")
        gsutil.copy(timestep_path, tmpdir)

        full_hash = utils.calc_directory_md5(os.path.join(tmpdir, timestep))

        checksum_path = os.path.join(tmpdir, "target-C48-checksum")
        with open(checksum_path, "w") as f:
            f.write(full_hash)
            logger.debug(f"Checksum saved to {checksum_path}")

        logger.info(f"Uploading to: {checksum_folder_destination}")
        gsutil.copy_directory_contents(tmpdir, checksum_folder_destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    src_data = "gs://vcm-ml-data/2020-01-16-X-SHiELD-2019-12-02-pressure-coarsened-rundirs/restarts/C48/"
    timestep = "20160801.001500"
    chksum_dst = "gs://vcm-ml-data/fv3net-testing-data/coarsen-timesteps/target-C48"

    upload_timestep_files_and_checksum(src_data, timestep, chksum_dst)
