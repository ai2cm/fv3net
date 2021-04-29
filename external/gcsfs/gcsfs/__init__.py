from ._version import get_versions

__version__ = '2021.4.0'
del get_versions
from .core import GCSFileSystem
from .mapping import GCSMap

__all__ = ["GCSFileSystem", "GCSMap"]
