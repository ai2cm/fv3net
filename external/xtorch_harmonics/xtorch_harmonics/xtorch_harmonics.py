import warnings

import numpy as np
import torch
import torch_harmonics
import xarray as xr

from typing import Hashable, Literal, TypeVar


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
LON_DIM = "grid_xt"
LAT_DIM = "grid_yt"

T_XarrayObject = TypeVar("T_XarrayObject", xr.DataArray, xr.Dataset)
T_Grid = Literal[EQUIANGULAR_GRID, LEGENDRE_GAUSS_GRID, LOBATTO_GRID]


def compute_quadrature_latitudes(n_lat: int, grid: T_Grid) -> np.ndarray:
    if grid not in QUADRATURE_FUNCTIONS:
        raise ValueError(f"Unrecognized grid {grid!r}.")

    quadrature_function = QUADRATURE_FUNCTIONS[grid]
    nodes, _ = quadrature_function(n_lat)
    return np.rad2deg(np.arcsin(nodes))


def compute_quadrature_longitudes(n_lon: int) -> np.ndarray:
    bounds = np.linspace(0, 360, n_lon + 1, endpoint=True)
    return (bounds[:-1] + bounds[1:]) / 2


def _validate_quadrature_points(
    obj: T_XarrayObject, forward_grid: T_Grid, lat_dim: Hashable, lon_dim: Hashable
):
    n_lat = obj.sizes[lat_dim]
    n_lon = obj.sizes[lon_dim]

    expected_latitudes = compute_quadrature_latitudes(n_lat, forward_grid)
    expected_longitudes = compute_quadrature_longitudes(n_lon)

    try:
        if lat_dim in obj.coords:
            np.testing.assert_allclose(obj[lat_dim].data, expected_latitudes)
            # TODO: could warn or raise if lat_dim not in obj.coords.
    except AssertionError:
        raise ValueError(
            f"Latitude coordinate {obj[lat_dim]} does not match the expected "
            f"quadrature points for the provided forward grid {forward_grid!r}."
        )

    try:
        if lon_dim in obj.coords:
            # TODO: I suppose for this check we really only need to ensure that
            # points are equally spaced and global.  Whether they exactly align
            # with a regular grid from 0 to 360 degrees E is not necessarily
            # relevant.
            np.testing.assert_allclose(obj[lon_dim].data, expected_longitudes)
            # TODO: could warn or raise if lon_dim not in obj.coords.
    except AssertionError:
        raise ValueError(
            f"Longitude coordinate {obj[lon_dim]} does not match the expected "
            f"quadrature points for the provided forward grid {forward_grid!r}."
        )


def _roundtrip_numpy(
    array: np.array, forward_grid: T_Grid, inverse_grid: T_Grid
) -> np.ndarray:
    *_, n_lat, n_lon = array.shape
    tensor = torch.tensor(array).type(torch.double)
    forward_transform = torch_harmonics.RealSHT(n_lat, n_lon, grid=forward_grid)
    inverse_transform = torch_harmonics.InverseRealSHT(n_lat, n_lon, grid=inverse_grid)
    roundtripped = inverse_transform(forward_transform(tensor))
    return np.array(roundtripped.type(torch.float))


def _roundtrip_dataarray(
    da: xr.DataArray,
    forward_grid: T_Grid,
    inverse_grid: T_Grid,
    lat_dim: Hashable,
    lon_dim: Hashable,
) -> xr.DataArray:
    # Ensure the DataArray is chunked contiguously in the horizontal, which is
    # required for a 2D spatial transform.
    horizontal_dims = {lat_dim, lon_dim}
    da = da.chunk({dim: -1 for dim in horizontal_dims})

    result = xr.apply_ufunc(
        _roundtrip_numpy,
        da,
        input_core_dims=[[lat_dim, lon_dim]],
        output_core_dims=[[lat_dim, lon_dim]],
        dask="parallelized",
        output_dtypes=[np.float32],
        keep_attrs=True,
        kwargs={"forward_grid": forward_grid, "inverse_grid": inverse_grid},
    )

    # Restore dimension order to match input DataArray
    return result.transpose(*da.dims)


def _roundtrip_dataset(
    ds: xr.Dataset,
    forward_grid: T_Grid,
    inverse_grid: T_Grid,
    lat_dim: Hashable,
    lon_dim: Hashable,
) -> xr.Dataset:
    horizontal_dims = {lon_dim, lat_dim}
    results = []
    for da in ds.data_vars.values():
        if not horizontal_dims.issubset(da.dims):
            results.append(da)
        else:
            result = _roundtrip_dataarray(
                da, forward_grid, inverse_grid, lat_dim, lon_dim
            )
            results.append(result)
    return xr.merge(results).assign_attrs(ds.attrs)


def roundtrip(
    obj: T_XarrayObject,
    forward_grid: T_Grid = LEGENDRE_GAUSS_GRID,
    inverse_grid: T_Grid = LEGENDRE_GAUSS_GRID,
    lat_dim: Hashable = LAT_DIM,
    lon_dim: Hashable = LON_DIM,
    unsafe: bool = False,
) -> T_XarrayObject:
    """
    Filter data by transforming to spherical harmonic space and back.

    Internally this uses the spectral transforms defined in NVIDIA's
    torch_harmonics package.  If a Dataset is provided, only data variables
    which contain both the provided latitude and longitude dimensions will be
    filtered; other data variables will be left unchanged.

    Args:
        obj: xr.DataArray or xr.Dataset
        forward_grid: str
            Grid to assume in forward transform (default 'legendre-gauss').
            Options are 'equiangular', 'legendre-gauss', and 'lobatto'.
        inverse_grid: str (default 'legendre-gauss')
            Grid to assume in inverse transform (default 'legendre-gauss').
            Options are 'equiangular', 'legendre-gauss', and 'lobatto'.
        lat_dim: Hashable
            Name of latitude dimension (default 'grid_yt').
        lon_dim: Hashable
            Name of longitude dimension (default 'grid_xt').
        unsafe: bool (default False)
            Whether to turn off guardrails that check whether the input
            quadrature points are consistent with the forward_grid.

    Returns:
        xr.DataArray or xr.Dataset
    """
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

    if not unsafe:
        _validate_quadrature_points(obj, forward_grid, lat_dim, lon_dim)

    if isinstance(obj, xr.DataArray):
        roundtrip_function = _roundtrip_dataarray
    elif isinstance(obj, xr.Dataset):
        roundtrip_function = _roundtrip_dataset
    else:
        raise ValueError(f"obj must be a DataArray or Dataset; got {type(obj)}")

    roundtripped = roundtrip_function(obj, forward_grid, inverse_grid, lat_dim, lon_dim)

    if forward_grid != inverse_grid and lat_dim in obj.coords:
        warnings.warn(
            "Modifying latitude coordinate since forward_grid does not match "
            "inverse_grid.",
            UserWarning,
            stacklevel=2,
        )
        latitudes = compute_quadrature_latitudes(obj.sizes[lat_dim], inverse_grid)
        roundtripped = roundtripped.assign_coords({lat_dim: latitudes})
    return roundtripped
