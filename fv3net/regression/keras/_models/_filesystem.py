# keras has several routines which interact with file paths directly as opposed to
# filesystem objects, which means we need these wrappers so we can allow remote paths

import contextlib
import tempfile
import fsspec
import os


@contextlib.contextmanager
def put_dir(path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
        fs, _, _ = fsspec.get_fs_token_paths(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        # cannot use fs.put as it cannot merge directories
        _put_directory(tmpdir, path)


@contextlib.contextmanager
def get_dir(path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fs, _, _ = fsspec.get_fs_token_paths(path)
        # fsspec places the directory inside the tmpdir, as a subdirectory
        fs.get(path, tmpdir, recursive=True)
        yield tmpdir


def _put_directory(
    local_source_dir: str, dest_dir: str, fs: fsspec.AbstractFileSystem = None,
):
    """Copy the contents of a local directory to a local or remote directory.
    """
    if fs is None:
        fs, _, _ = fsspec.get_fs_token_paths(dest_dir)
    fs.makedirs(dest_dir, exist_ok=True)
    for token in os.listdir(local_source_dir):
        source = os.path.join(os.path.abspath(local_source_dir), token)
        dest = os.path.join(dest_dir, token)
        if os.path.isdir(source):
            _put_directory(source, dest, fs=fs)
        else:
            fs.put(source, dest)
