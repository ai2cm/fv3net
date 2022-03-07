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


def to_url(fs: fsspec.AbstractFileSystem, path: str):
    """Convert a filesystem and path into a URI

    Args:
        fs: a filesystem object
        path: a path without a leading "protocol", as returned by ``fs.ls`` for example.

    Examples:
        >>> import vcm.cloud.fsspec
        >>> import fsspec
        >>> fs = fsspec.filesystem('file')
        >>> vcm.cloud.fsspec.to_url(fs, 'some-path')
        'file://some-path'
    """
    if isinstance(fs.protocol, str):
        protocol = fs.protocol
    elif "gs" in fs.protocol:
        protocol = "gs"
    else:
        protocol = fs.protocol[0]

    return protocol + "://" + path.lstrip("/")


def copy(source: str, destination: str, content_type: str = None):
    """Copy between any two 'filesystems'. Do not use for large files.

    Args:
        source: Path to source file/object.
        destination: Path to destination.
        content_type: MIME-type for destination. Not applied for local filesystems.
    """
    with fsspec.open(source) as f_source:
        with fsspec.open(destination, "wb") as f_destination:
            shutil.copyfileobj(f_source, f_destination)

    if content_type is not None:
        fs = get_fs(destination)
        if not isinstance(fs, fsspec.implementations.local.LocalFileSystem):
            fs.setxattrs(destination, content_type=content_type)
