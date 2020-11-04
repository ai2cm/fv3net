#!/usr/bin/env python
"""Consolidate metadata recursively of all zarrs below a directory

Uses multithreading

Usage:

    consolidate_metadata.py gs://bucket/prefix

"""

import fsspec
import sys
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from zarr.meta import json_dumps, json_loads

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _get_metadata_fs(fs, root):
    keys_to_get = [".zgroup"]

    arrays = fs.ls(root)
    for var in arrays:
        relpath = os.path.relpath(var, start=root)
        if relpath not in [".zgroup", ".zmetadata", ".zattrs"]:
            for key in [".zarray", ".zgroup", ".zattrs"]:
                keys_to_get.append(os.path.join(relpath, key))

    urls = [os.path.join(root, key) for key in keys_to_get]

    def maybe_get(url):
        try:
            return json_loads(fs.cat(url))
        except FileNotFoundError:
            pass

    with ThreadPoolExecutor(max_workers=12) as pool:
        values = pool.map(maybe_get, urls)

    metadata_with_nan = dict(zip(keys_to_get, values))
    metadata = {key: val for key, val in metadata_with_nan.items() if val is not None}
    return {"zarr_consolidated_format": 1, "metadata": metadata}


def consolidate_metadata(fs, root):
    if root.rstrip("/").endswith(".zarr"):
        logger.info(f"Consolidating metadata of {root}")
        meta = _get_metadata_fs(fs, root)

        with fs.open(os.path.join(root, ".zmetadata"), "wb") as f:
            f.write(json_dumps(meta))
    elif fs.isdir(root):
        logger.info(f"Recursing {root}")
        dirs = fs.ls(root)
        for dir_ in dirs:
            consolidate_metadata(fs, dir_)


if __name__ == "__main__":
    url = sys.argv[1]
    fs, _, roots = fsspec.get_fs_token_paths(url)
    root = roots[0]
    consolidate_metadata(fs, root)
