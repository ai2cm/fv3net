from pathlib import Path
from typing import Optional
import subprocess
import logging

logger = logging.getLogger(__name__)


def extract_tarball_to_path(
    source_tar_path: Path,
    extract_to_dir: Optional[Path] = None,
) -> Path:

    logger.info(f'Extracting tar file {source_tar_path.name}')

    # with suffix [blank] removes file_ext and uses filename as untar dir
    if extract_to_dir is None:
        extract_to_dir = source_tar_path.with_suffix('')

    extract_to_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f'Destination directory for tar extraction: {extract_to_dir}')
    tar_commands = ['tar', '-xf', source_tar_path, '-C', extract_to_dir]
    subprocess.check_call(tar_commands)

    return extract_to_dir
