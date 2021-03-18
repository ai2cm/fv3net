from external.vcm.vcm.testing import regression_data
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


def test_regression_data_unchanged(regtest):
    """This checks that the integration with regtests works and that the
    checksum report hasn't change
    """
    a = xarray.DataArray([1], dims=["x"], coords={"x": [1]}, attrs={"hello": "world"})
    print(regression_data(a), file=regtest)


def test_regression_data_different_attrs():
    a = xarray.DataArray([1], dims=["x"], attrs={"some": "attr"})
    b = xarray.DataArray([1], dims=["x"], attrs={"another": "attr"})

    assert testing.regression_data(a) != testing.regression_data(b)
    assert testing.regression_data(a, attrs=False) == testing.regression_data(
        b, attrs=False
    )


def test_regression_data_different_coords():
    a = xarray.DataArray([1], dims=["x"], coords={"x": [1]})
    b = xarray.DataArray([1], dims=["x"], coords={"x": [2]})

    assert testing.regression_data(a) != testing.regression_data(b)
    assert testing.regression_data(a, coords=False) == testing.regression_data(
        b, coords=False
    )


def test_regression_data_different_array():
    a = xarray.DataArray([2], dims=["x"])
    b = xarray.DataArray([1], dims=["x"])

    assert testing.regression_data(a) != testing.regression_data(b)
