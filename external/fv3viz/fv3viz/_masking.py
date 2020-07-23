import numpy as np


def _mask_antimeridian_quads(lonb: np.ndarray, central_longitude: float):
    """ Computes mask of cubed-sphere tile grid quadrilaterals bisected by a
    projection system's antimeridian, in order to avoid cartopy plotting
    artifacts

    Args:
        lonb (np.ndarray):
            Array of grid edge longitudes, of dimensions (npy + 1, npx + 1,
            tile)
        central_longitude (float):
            Central longitude from which the antimeridian is computed

    Returns:
        mask (np.ndarray):
            Boolean array of grid centers, False = excluded, of dimensions
            (npy, npx, tile)


    Example:
        masked_array = np.where(
            mask_antimeridian_quads(lonb, central_longitude),
            array,
            np.nan
        )
    """
    antimeridian = (central_longitude + 180.0) % 360.0
    mask = np.full([lonb.shape[0] - 1, lonb.shape[1] - 1, lonb.shape[2]], True)
    for tile in range(6):
        tile_lonb = lonb[:, :, tile]
        tile_mask = mask[:, :, tile]
        for ix in range(tile_lonb.shape[0] - 1):
            for iy in range(tile_lonb.shape[1] - 1):
                vertex_indices = ([ix, ix + 1, ix, ix + 1], [iy, iy, iy + 1, iy + 1])
                vertices = tile_lonb[vertex_indices]
                if (
                    sum(_periodic_equal_or_less_than(vertices, antimeridian)) != 4
                    and sum(_periodic_greater_than(vertices, antimeridian)) != 4
                    and sum((_periodic_difference(vertices, antimeridian) < 90.0)) == 4
                ):
                    tile_mask[ix, iy] = False
        mask[:, :, tile] = tile_mask

    return mask


def _periodic_equal_or_less_than(x1, x2, period=360.0):
    """ Compute whether x1 is less than or equal to x2, where
    the difference between the two is the shortest distance on a periodic domain

    Args:
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional):
            Period of domain. Default 360 (degrees).

    Returns:
        Less_than_or_equal (Bool):
            Whether x1 is less than or equal to x2
    """
    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        np.where(x1 - x2 <= 0, True, False),
        np.where(
            x1 - x2 >= 0,
            np.where(x1 - (x2 + period) <= 0, True, False),
            np.where((x1 + period) - x2 <= 0, True, False),
        ),
    )


def _periodic_greater_than(x1, x2, period=360.0):
    """ Compute whether x1 is greater than x2, where
    the difference between the two is the shortest distance on a periodic domain

    Args:
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional):
            Period of domain. Default 360 (degrees).

    Returns:
        Greater_than (Bool):
            Whether x1 is greater than x2
    """
    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        np.where(x1 - x2 > 0, True, False),
        np.where(
            x1 - x2 >= 0,
            np.where(x1 - (x2 + period) > 0, True, False),
            np.where((x1 + period) - x2 > 0, True, False),
        ),
    )


def _periodic_difference(x1, x2, period=360.0):
    """ Compute difference between x1 and x2, where
    the difference is the shortest distance on a periodic domain

    Args:
        x1 (float), x2 (float):
            Values to be compared
        Period (float, optional):
            Period of domain. Default 360 (degrees).

    Returns:
        Difference (float):
            Difference between x1 and x2
    """
    return np.where(
        np.abs(x1 - x2) <= period / 2.0,
        x1 - x2,
        np.where(x1 - x2 >= 0, x1 - (x2 + period), (x1 + period) - x2),
    )
