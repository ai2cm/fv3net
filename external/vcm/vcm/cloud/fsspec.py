import fsspec
import backoff
import gcsfs
import logging

logging.getLogger('backoff')


class FSWithBackoff(gcsfs.GCSFileSystem):
    
    @backoff.on_exception(backoff.constant, (AssertionError, RuntimeError), max_tries=10)
    def cat(self, key):
        getter = lambda: super(gcsfs.GCSFileSystem, self).cat(key)
        return getter()
    


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


def get_fs_with_retry_cat(path, project = 'vcm-ml'):

    protocol = get_protocol(path)
    
    if protocol == 'gs':
        fs = FSWithBackoff(project)
    else: 
        fs = fsspec.filesystem(protocol)
    return fs
