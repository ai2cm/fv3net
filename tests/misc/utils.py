import glob
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def calc_directory_md5(source_dir: str) -> str:
    """
    Calculate the md5 checksum of a directory by combining md5 hashes of
    individual files and calculating the md5 of that
    """

    files = glob.glob(os.path.join(source_dir, "*"))
    files.sort()
    logger.debug(f"Number of files in checksum: {len(files):d}")

    logger.info(f"Calculating directory checksum for {source_dir}")
    intermediate_hashes = [md5_hash_file(curr_file) for curr_file in files]
    combined_hashes = "".join(intermediate_hashes).encode("utf-8")
    dir_checksum = hashlib.md5(combined_hashes).hexdigest()

    return dir_checksum


def md5_hash_file(filepath: str) -> str:
    """
    Calculate the md5 hash of a file.
    """
    with open(filepath, "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
        logger.debug(
            f"Intermediate hash for file: {Path(filepath).name}\n" f"\thash: {md5_hash}"
        )
        return md5_hash
