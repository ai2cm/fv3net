import xarray as xr

CLIMIT1: float = 1.0e-3
CLIMIT2: float = 5.0e-2


def gridcell_to_incloud_condensate(
    cloud_fraction: xr.DataArray,
    gridcell_cloud_condensate: xr.DataArray,
    climit1: float = CLIMIT1,
    climit2: float = CLIMIT2,
) -> xr.DataArray:
    """Convert gridcell-mean condensate to in-cloud condensate via cloud fraction.

    Follows GFS physics condensate normalization; see
    https://github.com/ai2cm/fv3gfs-fortran/blob/5e1553cc34727399f629b233c36f1b4e8d2d902d/FV3/gfsphysics/physics/radiation_clouds.f#L1920 # noqa: E501


    Args:
        cloud_fraction: array of dimensionless cloud fractions
        gridcell_cloud_condensate: array of gridcell-mean condensate mixing ratios
        climit1: scalar of smallest cloud fraction where in-cloud condensate will be
            computed; where cloud fraction is smaller than this, in-cloud and gridcell-
            mean condensate will be the same
        climit2: scalar of minimum cloud fraction to compute in-cloud condensate; for
            values smaller than this (but greater than `climit1`), in-cloud condensate
            will be a factor of 1.0 / climit2 of gridcell-mean condensate

    Returns:
        Array of in-cloud condensate mixing ratios

    """
    scaling_ratio = 1.0 / cloud_fraction.where(cloud_fraction > climit2, climit2)
    incloud_condensate = gridcell_cloud_condensate.where(
        cloud_fraction <= climit1, gridcell_cloud_condensate * scaling_ratio
    )
    return incloud_condensate


def incloud_to_gridcell_condensate(
    cloud_fraction: xr.DataArray,
    incloud_condensate: xr.DataArray,
    climit1: float = CLIMIT1,
    climit2: float = CLIMIT2,
) -> xr.DataArray:
    """Convert in-cloud condensate to gridcell-mean condensate via cloud fraction

    Follows GFS physics condensate normalization; see
    https://github.com/ai2cm/fv3gfs-fortran/blob/5e1553cc34727399f629b233c36f1b4e8d2d902d/FV3/gfsphysics/physics/radiation_clouds.f#L1920 # noqa: E501

    Args:
        cloud_fraction: array of dimensionless cloud fractions
        incloud_condensate: array of in-cloud condensate mixing ratios
        climit1: scalar of smallest cloud fraction where gridcell-mean condensate will be
            computed; where cloud fraction is smaller than this, gridcell-mean and in-cloud
            condensate will be the same
        climit2: scalar of minimum cloud fraction to compute gridcell-mean condensate; for
            values smaller than this (but greater than `climit1`), gridcell-mean condensate
            will be a factor of climit2 of in-cloud condensate

    Returns: array of gridcell-mean condensate mixing ratios"""
    rectified_cloud_fraction = cloud_fraction.where(cloud_fraction > climit2, climit2)
    gridcell_condensate = incloud_condensate.where(
        cloud_fraction <= climit1, incloud_condensate * rectified_cloud_fraction
    )
    return gridcell_condensate
