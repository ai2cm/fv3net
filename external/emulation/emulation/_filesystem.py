# keras has several routines which interact with file paths directly as opposed to
# filesystem objects, which means we need these wrappers so we can allow remote paths

import contextlib
import tempfile
import fsspec


@contextlib.contextmanager
def get_dir(path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        fs, _, _ = fsspec.get_fs_token_paths(path)
        # fsspec places the directory inside the tmpdir, as a subdirectory
        fs.get(path, tmpdir, recursive=True)
        yield tmpdir
