import fsspec
import backoff
import gcsfs


class FSWithBackoff(gcsfs.GCSFileSystem):
    
    @backoff.on_exception(backoff.expo, AssertionError)
    def cat(self, key):
        getter = lambda: super(GCSFileSystem, self).cat(key)
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


def get_fs_with_retry_cat(path):

    protocol = get_protocol(path)
    
    if protocol == 'gs':
        fs = FSWithBackoff('vcm-ml')
    else: 
        fs = fsspec.filesystem(protocol)
    return fs
