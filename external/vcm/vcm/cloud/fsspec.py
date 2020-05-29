import fsspec  # type: ignore


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
