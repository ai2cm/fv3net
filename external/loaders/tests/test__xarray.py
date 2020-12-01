import doctest
import loaders.mappers._xarray

# ensure that imported path exists
from loaders.mappers import XarrayMapper  # noqa


def test_xarray_wrapper_doctests():
    doctest.testmod(loaders.mappers._xarray, raise_on_error=True)
