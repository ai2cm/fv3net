import cftime
import xarray as xr
import numpy as np
import pytest

from runtime.derived_state import DerivedFV3State, FV3StateMapper
import fv3gfs.util


class MockFV3GFS:
    def __init__(self, lon_units="degrees", lat_units="degrees"):
        self.set_state_called = False
        self.lon_units = lon_units
        self.lat_units = lat_units

    def get_state(self, names):
        # need this for the data to remain unchanged for equality tests
        np.random.seed(0)

        nx, ny = 10, 10
        lat_data = np.random.rand(ny, nx)
        lon_data = np.random.rand(ny, nx)

        if self.lat_units == "radians":
            lat_data = np.deg2rad(lat_data)
        if self.lon_units == "radians":
            lon_data = np.deg2rad(lon_data)

        lat = fv3gfs.util.Quantity(lat_data, dims=["y", "x"], units=self.lat_units)
        lon = fv3gfs.util.Quantity(lon_data, dims=["y", "x"], units=self.lon_units)
        lhtfl = fv3gfs.util.Quantity(
            np.random.rand(ny, nx), dims=["y", "x"], units="deg"
        )

        state = {
            "time": cftime.DatetimeJulian(2016, 1, 1),
            "latitude": lat,
            "longitude": lon,
            "lhtfl": lhtfl,
        }

        return {name: state[name] for name in names}

    def set_state_mass_conserving(self, data):
        self.set_state_called = True
        for key, value in data.items():
            assert isinstance(value, fv3gfs.util.Quantity)

    def get_diagnostic_by_name(self, diagnostic):
        return self.get_state([diagnostic])[diagnostic]


def test_DerivedFV3State():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    assert isinstance(getter["longitude"], xr.DataArray)


# test that function registered under DerivedMapping works
@pytest.mark.parametrize("lon_units", ["degrees", "radians"])
@pytest.mark.parametrize("lat_units", ["degrees", "radians"])
def test_DerivedFV3State_cos_zenith(lon_units, lat_units):
    result_fv3gfs = MockFV3GFS(lon_units=lon_units, lat_units=lat_units)
    expected_fv3gfs = MockFV3GFS()
    result_getter = DerivedFV3State(result_fv3gfs)
    expected_getter = DerivedFV3State(expected_fv3gfs)

    result = result_getter["cos_zenith_angle"]
    expected = expected_getter["cos_zenith_angle"]
    assert isinstance(result, xr.DataArray)
    assert "time" not in result.dims
    xr.testing.assert_identical(result, expected)


def test_DerivedFV3State_latent_heat_flux():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    output = getter["latent_heat_flux"]
    assert isinstance(output, xr.DataArray)


def test_DerivedFV3State_time_property():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    assert isinstance(getter.time, cftime.DatetimeJulian)


def test_DerivedFV3State_time_dataarray():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    assert isinstance(getter["time"], xr.DataArray)


def test_DerivedFV3State_setitem():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    item = xr.DataArray([1.0], dims=["x"], attrs={"units": "m"})
    # Check that data is passed to `MockFV3GFS.set_state` correctly
    getter["a"] = item
    assert fv3gfs.set_state_called


def test_FV3StateMapper():
    fv3gfs = MockFV3GFS()
    mapper = FV3StateMapper(fv3gfs)
    assert isinstance(mapper["latitude"], xr.DataArray)


def test_FV3StateMapper_alternate_keys():
    fv3gfs = MockFV3GFS()
    mapper = FV3StateMapper(fv3gfs, alternate_keys={"lon": "longitude"})
    np.testing.assert_array_almost_equal(mapper["lon"], mapper["longitude"])
