import cftime
import xarray as xr
import numpy as np

from runtime.derived_state import DerivedFV3State, FV3StateMapper
import fv3gfs.util


class MockFV3GFS:
    def __init__(self):
        self.set_state_called = False

    def get_state(self, names):
        # need this for the data to remain unchanged for equality tests
        np.random.seed(0)

        nx, ny = 10, 10

        lat = fv3gfs.util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")
        lon = fv3gfs.util.Quantity(np.random.rand(ny, nx), dims=["y", "x"], units="deg")
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
def test_DerivedFV3State_cos_zenith():
    fv3gfs = MockFV3GFS()
    getter = DerivedFV3State(fv3gfs)
    output = getter["cos_zenith_angle"]
    assert isinstance(output, xr.DataArray)


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
