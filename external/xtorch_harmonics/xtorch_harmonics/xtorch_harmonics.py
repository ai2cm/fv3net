import warnings

import numpy as np
import torch
import torch_harmonics
import xarray as xr

from typing import Callable, Hashable, Literal, TypeVar, Optional


EQUIANGULAR_GRID = "equiangular"
LEGENDRE_GAUSS_GRID = "legendre-gauss"
LOBATTO_GRID = "lobatto"
VALID_GRIDS = {
    (EQUIANGULAR_GRID, EQUIANGULAR_GRID),
    (LEGENDRE_GAUSS_GRID, LEGENDRE_GAUSS_GRID),
    (LOBATTO_GRID, LOBATTO_GRID),
    (EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID),
    (LEGENDRE_GAUSS_GRID, EQUIANGULAR_GRID),
}
QUADRATURE_FUNCTIONS = {
    EQUIANGULAR_GRID: torch_harmonics.quadrature.clenshaw_curtiss_weights,
    LEGENDRE_GAUSS_GRID: torch_harmonics.quadrature.legendre_gauss_weights,
    LOBATTO_GRID: torch_harmonics.quadrature.lobatto_weights,
}

T_XarrayObject = TypeVar("T_XarrayObject", xr.DataArray, xr.Dataset)
T_Grid = Literal["equiangular", "legendre-gauss", "lobatto"]


def compute_quadrature_latitudes(
    n_lat: int, grid: T_Grid, decreasing=False
) -> np.ndarray:
    if grid not in QUADRATURE_FUNCTIONS:
        raise ValueError(f"Unrecognized grid {grid!r}.")

    quadrature_function = QUADRATURE_FUNCTIONS[grid]
    nodes, _ = quadrature_function(n_lat)
    latitudes = np.rad2deg(np.arcsin(nodes))
    if decreasing:
        latitudes = latitudes[::-1]
    return latitudes


def compute_quadrature_longitudes(n_lon: int) -> np.ndarray:
    bounds = np.linspace(0, 360, n_lon + 1, endpoint=True)
    return (bounds[:-1] + bounds[1:]) / 2


def _validate_quadrature_longitudes(lon: np.ndarray) -> None:
    """Assert that longitudes are equally spaced and global.

    This validation function permits longitude coordinates that are increasing
    or decreasing, and longitude coordinates whose bounds start at values other
    than zero degrees.
    """
    (n_lon,) = lon.shape
    dlons = np.diff(lon)
    dlon = dlons[0]
    assert np.isclose(dlons, dlon).all(), (
        f"Longitude coordinate {lon} is not equally spaced, and must be equally "
        f"spaced for a valid roundtrip."
    )

    span = np.abs(dlon * n_lon)
    assert np.isclose(span, 360.0), (
        f"Longitude coordinate {lon} does not span 360 degrees, and must span "
        f"360 degrees for a valid roundtrip."
    )


def _validate_quadrature_latitudes(lat: np.ndarray, forward_grid: T_Grid) -> None:
    """Assert that latitudes match the expected quadrature latitudes for the
    given forward_grid.

    This validation function permits that latitudes be monotonically increasing
    or monotonically decreasing, both of which enable valid roundtrips.
    """
    (n_lat,) = lat.shape
    expected_increasing = compute_quadrature_latitudes(n_lat, forward_grid)
    expected_decreasing = compute_quadrature_latitudes(
        n_lat, forward_grid, decreasing=True
    )

    assert np.allclose(lat.data, expected_increasing) or np.allclose(
        lat.data, expected_decreasing
    ), (
        f"Latitude coordinate {lat} does not match the expected quadrature points "
        f"for the provided forward grid {forward_grid!r}."
    )


def _validate_quadrature_points(
    obj: T_XarrayObject, forward_grid: T_Grid, lat_dim: Hashable, lon_dim: Hashable
):
    if lat_dim in obj.coords:
        _validate_quadrature_latitudes(obj[lat_dim].values, forward_grid)
    else:
        warnings.warn(
            "No latitude coordinate exists; proceeding without validating quadrature "
            "points along the latitude dimension.",
            UserWarning,
            stacklevel=2,
        )

    if lon_dim in obj.coords:
        _validate_quadrature_longitudes(obj[lon_dim].values)
    else:
        warnings.warn(
            "No longitude coordinate exists; proceeding without validating quadrature "
            "points along the longitude dimension.",
            UserWarning,
            stacklevel=2,
        )


def _roundtrip_filter_numpy(
    array: np.array,
    forward_grid: T_Grid,
    inverse_grid: T_Grid,
    fraction_modes_kept: Optional[float] = None,
) -> np.ndarray:
    *_, n_lat, n_lon = array.shape
    tensor = torch.tensor(array).type(torch.float)
    forward_transform = torch_harmonics.RealSHT(n_lat, n_lon, grid=forward_grid)
    inverse_transform = torch_harmonics.InverseRealSHT(n_lat, n_lon, grid=inverse_grid)
    if fraction_modes_kept is None:
        roundtripped = inverse_transform(forward_transform(tensor))
    else:
        sht = forward_transform(tensor)
        sht[..., round(n_lat * fraction_modes_kept) :, :] = 0
        roundtripped = inverse_transform(sht)
    return np.array(roundtripped).astype(array.dtype)


def _roundtrip_filter_dataarray(
    da: xr.DataArray,
    forward_grid: T_Grid,
    inverse_grid: T_Grid,
    lat_dim: Hashable,
    lon_dim: Hashable,
    fraction_modes_kept: Optional[float] = None,
) -> xr.DataArray:
    # Ensure the DataArray is chunked contiguously in the horizontal, which is
    # required for a 2D spatial transform.
    horizontal_dims = {lat_dim, lon_dim}
    da = da.chunk({dim: -1 for dim in horizontal_dims})

    result = xr.apply_ufunc(
        _roundtrip_filter_numpy,
        da,
        input_core_dims=[[lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim]],
        dask="parallelized",
        output_dtypes=[da.dtype],
        keep_attrs=True,
        kwargs={
            "forward_grid": forward_grid,
            "inverse_grid": inverse_grid,
            "fraction_modes_kept": fraction_modes_kept,
        },
    )

    # Restore dimension order to match input DataArray
    return result.transpose(*da.dims)


def _roundtrip_filter_dataset(
    ds: xr.Dataset,
    forward_grid: T_Grid,
    inverse_grid: T_Grid,
    lat_dim: Hashable,
    lon_dim: Hashable,
    fraction_modes_kept: Optional[float] = None,
) -> xr.Dataset:
    horizontal_dims = {lon_dim, lat_dim}
    results = []
    for da in ds.data_vars.values():
        if not horizontal_dims.issubset(da.dims):
            results.append(da)
        else:
            result = _roundtrip_filter_dataarray(
                da, forward_grid, inverse_grid, lat_dim, lon_dim, fraction_modes_kept
            )
            results.append(result)
    return xr.merge(results).assign_attrs(ds.attrs)


def _replace_latitude_coordinate(
    obj: T_XarrayObject,
    roundtripped: T_XarrayObject,
    lat_dim: Hashable,
    inverse_grid: T_Grid,
) -> T_XarrayObject:
    warnings.warn(
        "Modifying latitude coordinate since forward_grid does not match "
        "inverse_grid.",
        UserWarning,
        stacklevel=3,
    )
    n_lat = obj.sizes[lat_dim]
    decreasing = obj.indexes[lat_dim].is_monotonic_decreasing
    latitudes = compute_quadrature_latitudes(n_lat, inverse_grid, decreasing)
    return roundtripped.assign_coords({lat_dim: latitudes})


def roundtrip_filter(
    obj: T_XarrayObject,
    lat_dim: Hashable,
    lon_dim: Hashable,
    forward_grid: T_Grid = "legendre-gauss",
    inverse_grid: T_Grid = "legendre-gauss",
    unsafe: bool = False,
    fraction_modes_kept: Optional[float] = None,
) -> T_XarrayObject:
    """
    Filter data by transforming to spherical harmonic space and back.

    Internally this uses the spectral transforms defined in NVIDIA's
    torch_harmonics package.  If a Dataset is provided, only data variables
    which contain both the provided latitude and longitude dimensions will be
    filtered; other data variables will be left unchanged.

    Args:
        obj: xr.DataArray or xr.Dataset
        lat_dim: Hashable
            Name of latitude dimension.
        lon_dim: Hashable
            Name of longitude dimension.
        forward_grid: str
            Grid to assume in forward transform (default 'legendre-gauss').
            Options are 'equiangular', 'legendre-gauss', and 'lobatto'.
        inverse_grid: str (default 'legendre-gauss')
            Grid to assume in inverse transform (default 'legendre-gauss').
            Options are 'equiangular', 'legendre-gauss', and 'lobatto'.
        unsafe: bool (default False)
            Whether to turn off guardrails that check whether the input
            quadrature points are consistent with the forward_grid.
        fraction_modes_kept: Fraction of modes (degrees) of the spherical harmonic
            to retain in the roundtrip. Truncation starts with the highest degrees.
            If None, all modes are retained. Must be between 0 and 1. The number of
            modes kept is the product of fraction_modes_kept and the number of
            harmonic degrees on the grid, rounded to the nearest integer.

    Returns:
        xr.DataArray or xr.Dataset
    """
    roundtrip_filter_function: Callable
    if (forward_grid, inverse_grid) not in VALID_GRIDS:
        raise ValueError(
            f"Provided forward and inverse grids ({forward_grid!r}, {inverse_grid!r}) "
            f"are unrecognized or incompatible for a roundtrip."
        )

    horizontal_dims = {lat_dim, lon_dim}
    if not horizontal_dims.issubset(obj.dims):
        raise ValueError(
            f"Input DataArray must have specified lat_dim "
            f"({lat_dim!r}) and lon_dim ({lon_dim!r})"
        )

    if fraction_modes_kept is not None:
        if fraction_modes_kept < 0 or fraction_modes_kept > 1:
            raise ValueError(
                (
                    "fraction_modes_kept must be between 0 and 1, "
                    f"got {fraction_modes_kept}."
                )
            )

    if not unsafe:
        _validate_quadrature_points(obj, forward_grid, lat_dim, lon_dim)

    if isinstance(obj, xr.DataArray):
        roundtrip_filter_function = _roundtrip_filter_dataarray
    elif isinstance(obj, xr.Dataset):
        roundtrip_filter_function = _roundtrip_filter_dataset
    else:
        raise ValueError(f"obj must be a DataArray or Dataset; got {type(obj)}")

    roundtripped = roundtrip_filter_function(
        obj, forward_grid, inverse_grid, lat_dim, lon_dim, fraction_modes_kept
    )

    if forward_grid != inverse_grid and lat_dim in obj.coords:
        roundtripped = _replace_latitude_coordinate(
            obj, roundtripped, lat_dim, inverse_grid
        )

    return roundtripped
