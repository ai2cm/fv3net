import contextlib
import logging
import os
import tempfile
import threading
import time

import fsspec

logger = logging.getLogger("vcm.cloud.fsspec")


def get_protocol(path: str) -> str:
    """Return fsspec filesystem protocol corresponding to path

    Args:
        path: local or remote path

    Returns:
        "file" unless "://" exists in path, in which case returns part of
        path preceding "://"
    """
    if "://" in path:
        protocol = path.split("://")[0]
    else:
        protocol = "file"
    return protocol


def get_fs(path: str) -> fsspec.AbstractFileSystem:
    """Return fsspec filesystem object corresponding to path"""
    return fsspec.filesystem(get_protocol(path))


def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def _show_usage(dir_, frequency: int = 10):
    while True:
        time.sleep(frequency)
        if os.path.isdir(dir_):
            usage = get_size(dir_)
            logger.debug(f"Cache usage: {usage/1024/1024:.2f} MB")
        else:
            break


@contextlib.contextmanager
def cached_fs(path: str):
    if path.startswith("gs://"):
        with tempfile.TemporaryDirectory() as dir_:
            logger.info(f"Opening gcs filesystem. Local cache at {dir_}")

            thread = threading.Thread(target=_show_usage, args=(dir_,))
            thread.start()

            yield fsspec.filesystem(
                "filecache", target_protocol="gs", cache_storage=dir_
            )
            logger.info(f"Cleaning {dir_}")
    else:
        yield get_fs(path)
