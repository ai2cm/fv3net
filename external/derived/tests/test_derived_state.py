from datetime import datetime
import xarray as xr
import numpy as np

from derived import DerivedFV3State
import fv3gfs.util

import pytest


class MockFV3GFS:
    def __init__(self):
        self.set_state_called = False

    def get_state(self, names):
        # need this for the data to remain unchanged for equality tests
        np.random.seed(0)

        nx, ny = 10, 10

        lat = fv3gfs.util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")
        lon = fv3gfs.util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")

        state = {
            "time": datetime.now(),
            "latitude": lat,
            "longitude": lon,
        }

        return {name: state[name] for name in names}

    def set_state_mass_conserving(self, data):
        self.set_state_called = True
        for key, value in data.items():
            assert isinstance(value, fv3gfs.util.Quantity)


def test_DerivedFV3State():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    assert isinstance(getter["longitude"], xr.DataArray)


def test_DerivedFV3State_cos_zenith():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    output = getter["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


def test_DerivedFV3State_time():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    assert isinstance(getter.time, datetime)


def test_DerivedFV3State_setitem():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    item = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
    # Check that data is passed to `MockFV3GFS.set_state` correctly
    getter["a"] = item
    assert fv3gfs.set_state_called


def test_DerivedFV3State_getitem_time_raises():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    with pytest.raises(KeyError):
        getter["time"]


def test_DerivedFV3State_register():
    fv3gfs = MockFV3GFS()

    def xarray_func(arr):
        return arr * 1.25

    @DerivedFV3State.register("mock")
    def mock(self):
        return xarray_func(self["latitude"])

    getter = DerivedFV3State(fv3gfs)

    latitude = getter["latitude"]
    expected = xarray_func(latitude)

    output = getter["mock"]
    assert isinstance(output, xr.DataArray)
    xr.testing.assert_equal(output, expected)
