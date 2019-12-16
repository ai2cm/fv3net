import xarray as xr
import xgcm
from . import constants

# none of the connecitons are "reversed" in the xgcm parlance
FV3_FACE_CONNECTIONS = {
    "tile": {
        1: {"x": ((5, "y", 1), (2, "x", 1)), "y": ((6, "y", 1), (3, "x", 1))},
        2: {"x": ((1, "x", 1), (4, "y", 1)), "y": ((6, "x", 1), (3, "y", 1))},
        3: {"x": ((1, "y", 1), (4, "x", 1)), "y": ((2, "y", 1), (5, "x", 1))},
        4: {"x": ((3, "x", 1), (6, "y", 1)), "y": ((2, "x", 1), (5, "y", 1))},
        5: {"x": ((3, "y", 1), (6, "x", 1)), "y": ((4, "y", 1), (1, "x", 1))},
        6: {"x": ((5, "x", 1), (2, "y", 1)), "y": ((4, "x", 1), (1, "y", 1))},
    }
}


def _validate_tile_coord(ds: xr.Dataset):

    if "tile" not in ds.coords:
        raise ValueError("The input Dataset must have a `tile` coordinate.")

    if not set(ds.tile) == {1, 2, 3, 4, 5, 6}:
        raise ValueError("`tile` coordinate must contain each of [1, 2, 3, 4, 5, 6]")


def create_fv3_grid(
    ds: xr.Dataset,
    x_center: str = constants.COORD_X_CENTER,
    x_outer: str = constants.COORD_X_OUTER,
    y_center: str = constants.COORD_Y_CENTER,
    y_outer: str = constants.COORD_Y_OUTER,
) -> xgcm.Grid:
    """Create an XGCM_ grid from a dataset of FV3 tile data


    Args:
        ds: dataset with a valid tiles dimension. The tile dimension must have a 
            corresponding coordinate. To follow GFDL's convention, this coordinate 
            should start with 1. You can make it like this::

                ds = ds.assign_coords(tile=np.arange(1, 7))

        x_center (optional): the dimension name for the x edges
        x_outer (optional): the dimension name for the x edges
        y_center (optional): the dimension name for the y edges
        y_outer (optional): the dimension name for the y edges

    Returns:
        an xgcm grid object. This object can be used to interpolate and differentiate 
        cubed sphere data, please see the XGCM_ documentation for more information.

    Notes:
        See this notebook_ for usage.


    .. _XGCM: https://xgcm.readthedocs.io/en/latest/
    .. _notebook: https://github.com/VulcanClimateModeling/explore/blob/master/noahb/2019-12-06-XGCM.ipynb # noqa

    """

    _validate_tile_coord(ds)

    coords = {
        "x": {"center": x_center, "outer": x_outer},
        "y": {"center": y_center, "outer": y_outer},
    }
    return xgcm.Grid(ds, coords=coords, face_connections=FV3_FACE_CONNECTIONS)
