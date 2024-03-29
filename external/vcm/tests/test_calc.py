import cftime
import numpy as np
import pytest
import xarray as xr
from vcm.calc.thermo.local import (
    internal_energy,
    relative_humidity,
    specific_humidity_from_rh,
)
from vcm.calc.thermo.vertically_dependent import (
    pressure_at_interface,
    height_at_interface,
    _interface_to_midpoint,
    dz_and_top_to_phis,
    _add_coords_to_interface_variable,
    mass_streamfunction,
)
from vcm.calc.calc import local_time, weighted_average, vertical_tapering_scale_factors
from vcm.cubedsphere.constants import COORD_Z_CENTER, COORD_Z_OUTER
from vcm.calc.thermo.constants import _GRAVITY, _RDGAS, _RVGAS
import vcm


@pytest.mark.parametrize("toa_pressure", [0, 5])
def test_pressure_on_interface(toa_pressure):
    delp = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    pressure = pressure_at_interface(delp, toa_pressure=toa_pressure)
    xr.testing.assert_allclose(
        pressure.diff(COORD_Z_OUTER), delp.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )
    np.testing.assert_allclose(pressure.isel({COORD_Z_OUTER: 0}).values, toa_pressure)


@pytest.mark.parametrize("phis_value", [0, 5])
def test_height_on_interface(phis_value):
    dz = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_CENTER])
    phis = xr.DataArray(phis_value)
    height = height_at_interface(dz, phis)
    xr.testing.assert_allclose(
        height.diff(COORD_Z_OUTER), dz.rename({COORD_Z_CENTER: COORD_Z_OUTER})
    )
    np.testing.assert_allclose(
        height.isel({COORD_Z_OUTER: -1}).values, phis_value / _GRAVITY
    )


def test__interface_to_midpoint():
    interface = xr.DataArray(np.arange(1, 10), dims=[COORD_Z_OUTER])
    midpoint = _interface_to_midpoint(interface)
    expected = xr.DataArray(np.arange(1.5, 9), dims=[COORD_Z_CENTER])
    xr.testing.assert_allclose(midpoint, expected)


@pytest.mark.parametrize(
    "da_coords, da_out_expected_coords",
    [
        ({"x": range(5)}, {"x": range(5)}),
        ({"z": range(10), "x": range(5)}, {"x": range(5)}),
    ],
)
def test__add_coords_to_interface_variable(da_coords, da_out_expected_coords):
    da = xr.DataArray(np.ones((10, 5)), dims=["z", "x"], coords=da_coords)
    dv = xr.Variable(["z", "x"], np.ones((11, 5)))
    da_out = _add_coords_to_interface_variable(dv, da, dim_center="z")
    da_out_expected = xr.DataArray(
        np.ones((11, 5)), dims=["z", "x"], coords=da_out_expected_coords
    )
    xr.testing.assert_allclose(da_out, da_out_expected)


def test_dz_and_top_to_phis():
    top = 10.0
    dz = np.arange(-4, -1)
    dza = xr.DataArray(dz, dims=[COORD_Z_CENTER])
    phis = dz_and_top_to_phis(top, dza)
    np.testing.assert_allclose(phis.values / _GRAVITY, top + np.sum(dz))


def test_solar_time():
    t = xr.DataArray(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 0, 0),
            cftime.DatetimeJulian(2020, 1, 1, 6, 0),
            cftime.DatetimeJulian(2020, 1, 1, 6, 0),
        ],
        dims=["x"],
        coords={"x": range(6)},
    )
    lon = xr.DataArray([0, 180, 270, 360, 0, 270], dims=["x"], coords={"x": range(6)})
    ds_solar_test = xr.Dataset({"initialization_time": t, "lon": lon})
    assert np.allclose(local_time(ds_solar_test), [0, 12, 18, 0, 6, 0])


def test_mass_streamfunction():
    latitude = np.array([-75, -25, 25, 75])
    pressure = np.array([5000, 10000, 30000, 75000, 90000])
    wind = xr.DataArray(
        np.reshape(np.arange(0, 40, 2), (5, 4)),
        coords={"pressure": pressure, "latitude": latitude},
        dims=["pressure", "latitude"],
    )
    psi = mass_streamfunction(wind)
    assert dict(psi.sizes) == {"pressure": 5, "latitude": 4}
    np.testing.assert_equal(psi.pressure.values, pressure)


def test_internal_energy():
    temperature = xr.DataArray(
        np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],
    )
    energy = internal_energy(temperature)
    assert energy.shape == temperature.shape


@pytest.mark.parametrize("celsius, rh", [(26, 0.5), (14.77, 1.0)])
def test_relative_humidity(celsius, rh):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """

    rho = 1.1781
    p = 1018_00
    e = 16_79.30

    q = _RDGAS / _RVGAS * e / (p - e)

    T = celsius + 273.15
    ans = relative_humidity(T, q, rho)

    assert pytest.approx(rh, rel=0.03) == ans


@pytest.mark.parametrize("t", [200, 250, 300])
@pytest.mark.parametrize("rh", [0, 0.5, 1.0])
@pytest.mark.parametrize("rho", [1.2, 1e-4])
def test_specific_humidity(t, rh, rho):
    """
    Compare withh https://www.omnicalculator.com/physics/air-density
    """
    q = specific_humidity_from_rh(t, rh, rho)
    rh_round_trip = relative_humidity(t, q, rho)

    assert pytest.approx(rh) == rh_round_trip


def test_moist_static_energy_tendency():
    Q1 = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Q2 = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Qm = vcm.moist_static_energy_tendency(Q1, Q2)
    assert Qm.shape == Q1.shape


def test_temperature_tendency():
    Qm = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Q2 = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Q1 = vcm.temperature_tendency(Qm, Q2)
    assert Qm.shape == Q1.shape


def test_round_trip_mse_temperature_tendency():
    Q1 = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Q2 = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "y"],)
    Qm = vcm.moist_static_energy_tendency(Q1, Q2)
    round_tripped_Q1 = vcm.temperature_tendency(Qm, Q2)
    xr.testing.assert_allclose(Q1, round_tripped_Q1)


def test_mass_cumsum():
    da = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "z"])
    delp = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "z"])
    out = vcm.mass_cumsum(da, delp, dim="z")
    assert set(out.dims) == {"x", "z"}
    assert out.shape == (5, 4)


def test_mass_divergence():
    da = xr.DataArray(np.reshape(np.arange(0, 50, 2), (5, 5)), dims=["x", "phalf"])
    delp = xr.DataArray(np.reshape(np.arange(0, 40, 2), (5, 4)), dims=["x", "pfull"])
    out = vcm.mass_divergence(da, delp, dim_center="pfull", dim_interface="phalf")
    assert set(out.dims) == {"x", "pfull"}
    assert out.shape == (5, 4)


def test_relative_humidity_from_pressure():
    # values taken from https://www.omnicalculator.com/physics/air-density
    pressure = 1e5
    temperature = 300
    e = 1412
    q = 0.622 * e / (pressure - 0.378 * e)
    expected_rh = 0.4
    rh = vcm.relative_humidity_from_pressure(temperature, q, pressure)
    assert pytest.approx(expected_rh, rel=1e-3) == rh


da = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])
da_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])
da_some_nans = xr.DataArray([1, 2, 3, np.nan], dims=["z"])
ds = xr.Dataset({"a": da})
weights = xr.DataArray([0.5, 0.5, 1, 1], dims=["z"])
weights_some_nans = xr.DataArray([0.5, 0.5, np.nan, 1], dims=["z"])
weights_nans = xr.DataArray(np.full((4,), np.nan), dims=["z"])


@pytest.mark.parametrize(
    "da,weights,dims,expected",
    [
        (da, weights, "z", xr.DataArray(17.0 / 6.0)),
        (ds, weights, "z", xr.Dataset({"a": xr.DataArray(17.0 / 6.0)})),
        (da_nans, weights, "z", xr.DataArray(np.nan)),
        (da, weights_nans, "z", xr.DataArray(np.nan)),
        (da, weights_some_nans, "z", xr.DataArray((0.5 + 1 + 4) / (0.5 + 0.5 + 1))),
        (da_some_nans, weights, "z", xr.DataArray((0.5 + 1 + 3) / (0.5 + 0.5 + 1))),
    ],
)
def test_weighted_average(da, weights, dims, expected):
    xr.testing.assert_allclose(weighted_average(da, weights, dims), expected)


def test_weighted_averaged_no_dims():

    da = xr.DataArray([[[np.arange(1.0, 5.0)]]], dims=["tile", "y", "x", "z"])
    weights = xr.DataArray([[[[0.5, 0.5, 1, 1]]]], dims=["tile", "y", "x", "z"])
    expected = xr.DataArray(np.arange(1.0, 5.0), dims=["z"])

    xr.testing.assert_allclose(weighted_average(da, weights), expected)


@pytest.mark.parametrize(
    "cutoff, rate, expected",
    [
        (0, 5, np.array([1.0, 1.0, 1.0, 1.0])),
        (2, 5, np.array([np.exp(-2.0 / 5), np.exp(-1.0 / 5), 1.0, 1.0])),
    ],
)
def test_vertical_tapering_scale_factors(cutoff, rate, expected):
    x = np.ones(4)
    np.testing.assert_array_almost_equal(
        x * vertical_tapering_scale_factors(4, cutoff=cutoff, rate=rate), expected
    )


def test_weighted_averaged_keeps_attrs():
    da = xr.DataArray(
        [[[np.arange(1.0, 5.0)]]],
        dims=["tile", "y", "x", "z"],
        attrs={"units": "unit_name", "other": "foo"},
    )
    weights = xr.DataArray([[[[0.5, 0.5, 1, 1]]]], dims=["tile", "y", "x", "z"])
    expected = xr.DataArray(
        np.arange(1.0, 5.0), dims=["z"], attrs={"units": "unit_name", "other": "foo"}
    )
    xr.testing.assert_identical(weighted_average(da, weights), expected)
