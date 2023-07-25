import numpy as np
import pandas as pd
import pytest
import scipy.special
import xarray as xr

from xtorch_harmonics.xtorch_harmonics import (
    EQUIANGULAR_GRID,
    LEGENDRE_GAUSS_GRID,
    LOBATTO_GRID,
    VALID_GRIDS,
)
from xtorch_harmonics.xtorch_harmonics import (
    _validate_quadrature_latitudes,
    _validate_quadrature_longitudes,
    compute_quadrature_latitudes,
    compute_quadrature_longitudes,
)
from xtorch_harmonics.xtorch_harmonics import roundtrip


HARMONIC_DIM = "harmonic"
N_LAT, N_LON = 9, 18
LON_DIM, LAT_DIM = "grid_xt", "grid_yt"


def real_spherical_harmonic(lat, lon, m, n):
    dims = lat.dims
    phi = np.deg2rad(90 - lat)
    theta = np.deg2rad(lon).transpose(*dims)

    result = xr.apply_ufunc(
        scipy.special.sph_harm,
        m,
        n,
        theta,
        phi,
        input_core_dims=[[], [], dims, dims],
        output_core_dims=[dims],
    )
    return result.real


def horizontal_grid(grid, n_lat=N_LAT, n_lon=N_LON, decreasing_latitude=False):
    lat = compute_quadrature_latitudes(n_lat, grid, decreasing=decreasing_latitude)
    lon = compute_quadrature_longitudes(n_lon)
    return lat, lon


def constant_dataarray(grid, lat_dim, lon_dim, decreasing_latitude=False, name="foo"):
    lat, lon = horizontal_grid(grid, decreasing_latitude=decreasing_latitude)
    data = np.ones((N_LAT, N_LON))
    return xr.DataArray(data, dims=[lat_dim, lon_dim], coords=[lat, lon], name=name)


def real_spherical_harmonic_dataarray(grid, lat_dim, lon_dim, name="foo"):
    lat, lon = horizontal_grid(grid)
    lat = xr.DataArray(lat, dims=[lat_dim], coords=[lat])
    lon = xr.DataArray(lon, dims=[lon_dim], coords=[lon])
    lat, lon = xr.broadcast(lat, lon)

    orders = [-1, 0, 1]
    degrees = [1, 2, 3]
    harmonics = pd.MultiIndex.from_product([orders, degrees], names=["m", "n"])

    dataarrays = []
    for m, n in harmonics:
        da = real_spherical_harmonic(lat, lon, m, n)
        dataarrays.append(da)

    return (
        xr.concat(dataarrays, dim=HARMONIC_DIM)
        .assign_coords({HARMONIC_DIM: harmonics})
        .unstack(HARMONIC_DIM)
        .rename(name)
    )


@pytest.mark.parametrize(("forward_grid", "inverse_grid"), VALID_GRIDS)
@pytest.mark.parametrize("decreasing_latitude", [False, True])
def test_roundtrip_constant_dataarray(forward_grid, inverse_grid, decreasing_latitude):
    da = constant_dataarray(
        forward_grid, LAT_DIM, LON_DIM, decreasing_latitude=decreasing_latitude
    )

    if forward_grid != inverse_grid:
        with pytest.warns(UserWarning, match="Modifying latitude coordinate"):
            result = roundtrip(da, LAT_DIM, LON_DIM, forward_grid, inverse_grid,)
    else:
        result = roundtrip(da, LAT_DIM, LON_DIM, forward_grid, inverse_grid)

    expected_latitude = compute_quadrature_latitudes(
        da.sizes[LAT_DIM], inverse_grid, decreasing_latitude
    )
    expected = da.assign_coords({LAT_DIM: expected_latitude})
    xr.testing.assert_allclose(result, expected)
    assert da.indexes[LAT_DIM].is_monotonic_decreasing == decreasing_latitude


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_roundtrip_dataarray_preserves_dtype(dtype):
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM).astype(dtype)
    result = roundtrip(da, LAT_DIM, LON_DIM)
    assert result.dtype == da.dtype


@pytest.mark.parametrize(
    ("grid", "rtol"),
    [
        (EQUIANGULAR_GRID, 1e-1),  # Not clear why this requires such a large tolerance.
        (LEGENDRE_GAUSS_GRID, 1e-5),
        (LOBATTO_GRID, 1e-5),
    ],
)
def test_roundtrip_real_spherical_harmonic_dataarray(grid, rtol):
    # We expect spherical harmonics themselves to pass through approximately
    # unchanged if we use the same grid for the forward and inverse transforms.
    # This also tests the use of roundtrip on DataArrays with more than two
    # dimensions; da has dimensions ["grid_yt", "grid_xt", "m", "n"].
    da = real_spherical_harmonic_dataarray(grid, LAT_DIM, LON_DIM)
    result = roundtrip(da, LAT_DIM, LON_DIM, forward_grid=grid, inverse_grid=grid)
    xr.testing.assert_allclose(result, da, rtol=rtol)


@pytest.mark.parametrize(
    "chunks",
    [{LAT_DIM: N_LAT // 2, LON_DIM: N_LON // 3}, {"m": 1, "n": 2}],
    ids=["horizontal-chunks-eliminated", "non-horizontal-chunks-preserved"],
)
def test_roundtrip_dataarray_dask(chunks):
    da = real_spherical_harmonic_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM)
    da = da.chunk(chunks)
    result = roundtrip(da, LAT_DIM, LON_DIM)

    # Assert that all non-horizontal chunks are preserved, and the result is
    # contiguously chunked along horizontal dimensions.
    expected_chunks = []
    for dim in da.dims:
        axis = da.get_axis_num(dim)
        if dim not in {LAT_DIM, LON_DIM}:
            expected_chunks.append(da.chunks[axis])
        else:
            expected_chunks.append((da.sizes[dim],))
    assert result.chunks == tuple(expected_chunks)

    # Check that the result is computed accurately.
    xr.testing.assert_allclose(result, da)


def test_roundtrip_dataarray_keep_attrs():
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM)
    da = da.assign_attrs(bar="baz")
    result = roundtrip(da, LAT_DIM, LON_DIM)
    assert da.attrs == result.attrs


def test_roundtrip_dataarray_incompatible_forward_inverse_grids():
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM)
    with pytest.raises(ValueError, match="Provided forward and inverse grids"):
        roundtrip(
            da,
            LAT_DIM,
            LON_DIM,
            forward_grid=LEGENDRE_GAUSS_GRID,
            inverse_grid=LOBATTO_GRID,
        )


def test_roundtrip_dataarray_incomplete_horizontal_dims():
    lat_dim, lon_dim = "lat", "lon"
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, lat_dim, lon_dim)
    with pytest.raises(ValueError, match="Input DataArray must have"):
        roundtrip(da, LAT_DIM, LON_DIM)


@pytest.mark.parametrize(
    ("input_grid", "forward_grid", "inverse_grid"),
    [
        (EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID, EQUIANGULAR_GRID),
        (LEGENDRE_GAUSS_GRID, EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID),
        (LOBATTO_GRID, EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID),
    ],
)
@pytest.mark.parametrize("unsafe", [False, True])
def test_latitude_quadrature_point_validation(
    input_grid, forward_grid, inverse_grid, unsafe
):
    da = constant_dataarray(input_grid, LAT_DIM, LON_DIM)

    if unsafe:
        with pytest.warns(UserWarning, match="Modifying latitude"):
            roundtrip(da, LAT_DIM, LON_DIM, forward_grid, inverse_grid, unsafe=unsafe)
    else:
        with pytest.raises(AssertionError, match="Latitude coordinate"):
            roundtrip(da, LAT_DIM, LON_DIM, forward_grid, inverse_grid, unsafe=unsafe)


@pytest.mark.parametrize("unsafe", [False, True])
def test_longitude_quadrature_point_validation(unsafe):
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM)
    da = da.assign_coords({LON_DIM: np.sin(da[LON_DIM])})

    if unsafe:
        roundtrip(da, LAT_DIM, LON_DIM, unsafe=unsafe)
    else:
        with pytest.raises(AssertionError, match="Longitude coordinate"):
            roundtrip(da, LAT_DIM, LON_DIM, unsafe=unsafe)


@pytest.mark.parametrize("unsafe", [False, True])
def test_validation_warnings_without_coordinates(unsafe):
    da = constant_dataarray(LEGENDRE_GAUSS_GRID, LAT_DIM, LON_DIM)
    da = da.drop_vars([LAT_DIM, LON_DIM])
    if unsafe:
        roundtrip(da, LAT_DIM, LON_DIM, unsafe=unsafe)
    else:
        with pytest.warns(None) as record:
            roundtrip(da, LAT_DIM, LON_DIM, unsafe=unsafe)

        latitude_warning, longitude_warning = record
        assert issubclass(latitude_warning.category, UserWarning)
        assert issubclass(longitude_warning.category, UserWarning)
        assert "No latitude" in str(latitude_warning.message)
        assert "No longitude" in str(longitude_warning.message)


@pytest.mark.parametrize(
    "forward_grid", [EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID, LOBATTO_GRID]
)
def test__validate_quadrature_latitudes(forward_grid):
    lat = compute_quadrature_latitudes(N_LAT, forward_grid)
    _validate_quadrature_latitudes(lat, forward_grid)
    _validate_quadrature_latitudes(lat[::-1], forward_grid)

    with pytest.raises(AssertionError, match="Latitude coordinate"):
        _validate_quadrature_latitudes(lat[:-1], forward_grid)


@pytest.mark.parametrize(
    ("lon", "raises", "match"),
    [
        (np.array([45, 135, 225, 315]), False, None),
        (np.array([-135, -45, 45, 135]), False, None),
        (np.array([-135, -45, 45]), True, "span 360 degrees"),
        (np.array([-135, 45, 135]), True, "equally spaced"),
    ],
    ids=lambda x: str(x),
)
def test__validate_quadrature_longitudes(lon, raises, match):
    if raises:
        with pytest.raises(AssertionError, match=match):
            _validate_quadrature_longitudes(lon)
    else:
        _validate_quadrature_longitudes(lon)


def test_roundtrip_dataset():
    grid = LOBATTO_GRID
    foo = real_spherical_harmonic_dataarray(grid, LAT_DIM, LON_DIM)
    bar = foo.copy(deep=True).rename("bar").isel({LAT_DIM: 0})
    ds = xr.merge([foo, bar])
    ds = ds.assign_attrs(a="b")
    roundtripped = roundtrip(
        ds, LAT_DIM, LON_DIM, forward_grid=grid, inverse_grid=grid,
    )

    # Check that foo was modified and bar was left unchanged.
    with pytest.raises(AssertionError):
        xr.testing.assert_identical(roundtripped.foo, ds.foo)

    xr.testing.assert_identical(roundtripped.bar, ds.bar)

    # Check that Dataset-level attributes are preserved in the roundtrip.
    assert ds.attrs == roundtripped.attrs
