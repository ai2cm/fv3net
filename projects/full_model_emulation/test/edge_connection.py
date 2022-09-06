import numpy as np
from pace.util.constants import PI
from vcm.catalog import catalog


def normalize_vector(np, *vector_components):
    scale = np.divide(
        1.0,
        np.sum(np.asarray([item ** 2.0 for item in vector_components]), axis=0) ** 0.5,
    )
    return np.asarray([item * scale for item in vector_components])


def normalize_xyz(xyz):
    # double transpose to broadcast along last dimension instead of first
    return (xyz.T / ((xyz ** 2).sum(axis=-1) ** 0.5).T).T


def lon_lat_to_xyz(lon, lat, np):
    """map (lon, lat) to (x, y, z)
    Args:
        lon: 2d array of longitudes
        lat: 2d array of latitudes
        np: numpy-like module for arrays
    Returns:
        xyz: 3d array whose last dimension is length 3 and indicates x/y/z value
    """
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    x, y, z = normalize_vector(np, x, y, z)
    if len(lon.shape) == 2:
        xyz = np.concatenate([arr[:, :, None] for arr in (x, y, z)], axis=-1)
    elif len(lon.shape) == 1:
        xyz = np.concatenate([arr[:, None] for arr in (x, y, z)], axis=-1)
    return xyz


def xyz_to_lon_lat(xyz, np):
    """map (x, y, z) to (lon, lat)
    Returns:
        xyz: 3d array whose last dimension is length 3 and indicates x/y/z value
        np: numpy-like module for arrays
    Returns:
        lon: 2d array of longitudes
        lat: 2d array of latitudes
    """
    xyz = normalize_xyz(xyz)
    # double transpose to index last dimension, regardless of number of dimensions
    x = xyz.T[0, :].T
    y = xyz.T[1, :].T
    z = xyz.T[2, :].T
    lon = 0.0 * x
    nonzero_lon = np.abs(x) + np.abs(y) >= 1.0e-10
    lon[nonzero_lon] = np.arctan2(y[nonzero_lon], x[nonzero_lon])
    negative_lon = lon < 0.0
    while np.any(negative_lon):
        lon[negative_lon] += 2 * PI
        negative_lon = lon < 0.0
    lat = np.arcsin(z)
    return lon, lat


def coarse_grid(nx: int):
    grid = catalog["grid/c48"].read()
    lat = grid.lat.load()
    lon = grid.lon.load()
    lat = lat.transpose("tile", "x", "y")
    lon = lon.transpose("tile", "x", "y")
    lat = lat.values.flatten() / 180 * np.pi
    lon = lon.values.flatten() / 180 * np.pi
    xyz = lon_lat_to_xyz(lon, lat, np)
    xyz = xyz.reshape((6, 48, 48, 3))
    coarse_level = 48 // nx
    for coarse in range(1, int(np.log2(coarse_level)) + 1):
        xyz = (
            xyz[:, 1::2, :-1:2]
            + xyz[:, :-1:2, 1::2]
            + xyz[:, 1::2, 1::2]
            + xyz[:, :-1:2, :-1:2]
        )
    xyz = xyz.reshape((6 * nx * nx, 3))
    lon_new, lat_new = xyz_to_lon_lat(xyz, np)
    lon_new = lon_new.reshape((6, nx, nx))
    lat_new = lat_new.reshape((6, nx, nx))
    return lon_new, lat_new