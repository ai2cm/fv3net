import warnings
import pytest

from vcm import testing
import xarray


def test_no_warning():
    with pytest.raises(AssertionError):
        with testing.no_warning(None):
            warnings.warn("Warning")


def test_checksum_dataarray(regtest):
    """If these checksums fail then some changed probably happened in
    joblib.hash
    """
    array = xarray.DataArray([1], dims=["x"])
    print(testing.checksum_dataarray(array), file=regtest)


def test_checksum_dataarray_mapping(regtest):
    """If these checksums fail then some changed probably happened in
    joblib.hash
    """
    ds = xarray.Dataset({"one": ("x", [1]), "two": ("x", [2])})
    print(testing.checksum_dataarray_mapping(ds), file=regtest)
