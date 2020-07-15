from datetime import datetime
import xarray as xr
import numpy as np

import runtime
import fv3util

import pytest


class MockFV3GFS:
    def get_state(self, names):
        # need this for the data to remain unchanged for equality tests
        np.random.seed(0)

        nx, ny = 10, 10

        lat = fv3util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")
        lon = fv3util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")

        state = {
            "time": datetime.now(),
            "latitude": lat,
            "longitude": lon,
        }

        return {name: state[name] for name in names}


def test_DerivedVariableGetter():
    fv3gfs = MockFV3GFS()
    getter = runtime.DerivedVariableGetter(fv3gfs)
    assert isinstance(getter["longitude"], xr.DataArray)


def test_DerivedVariableGetter_cos_zenith():
    fv3gfs = MockFV3GFS()
    getter = runtime.DerivedVariableGetter(fv3gfs)
    output = getter["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedVariableGetter_time():
    fv3gfs = MockFV3GFS()
    getter = runtime.DerivedVariableGetter(fv3gfs)
    assert isinstance(getter.time, datetime)


def test_DerivedVariableGetter_getitem_time_raises():
    fv3gfs = MockFV3GFS()
    getter = runtime.DerivedVariableGetter(fv3gfs)
    with pytest.raises(KeyError):
        getter["time"]


def test_DerivedVariableGetter_register():
    fv3gfs = MockFV3GFS()

    def xarray_func(arr):
        return arr * 1.25

    @runtime.DerivedVariableGetter.register("mock")
    def mock(self):
        return xarray_func(self["latitude"])

    getter = runtime.DerivedVariableGetter(fv3gfs)

    latitude = getter["latitude"]
    expected = xarray_func(latitude)

    output = getter["mock"]
    assert isinstance(output, xr.DataArray)
    xr.testing.assert_equal(output, expected)
