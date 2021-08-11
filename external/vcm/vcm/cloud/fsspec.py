import shutil
import fsspec


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


def copy(source: str, destination: str, content_type: str = None):
    """Copy between any two 'filesystems'. Do not use for large files."""
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)

    if content_type is not None:
        fs = get_fs(destination)
        fs.setxattrs(destination, content_type=content_type)
