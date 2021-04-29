import xarray as xr
import xgcm
from . import constants

# none of the connecitons are "reversed" in the xgcm parlance
FV3_FACE_CONNECTIONS = {
    "tile": {
        0: {
            "x": ((4, "y", False), (1, "x", False)),
            "y": ((5, "y", False), (2, "x", False)),
        },
        1: {
            "x": ((0, "x", False), (3, "y", False)),
            "y": ((5, "x", False), (2, "y", False)),
        },
        2: {
            "x": ((0, "y", False), (3, "x", False)),
            "y": ((1, "y", False), (4, "x", False)),
        },
        3: {
            "x": ((2, "x", False), (5, "y", False)),
            "y": ((1, "x", False), (4, "y", False)),
        },
        4: {
            "x": ((2, "y", False), (5, "x", False)),
            "y": ((3, "y", False), (0, "x", False)),
        },
        5: {
            "x": ((4, "x", False), (1, "y", False)),
            "y": ((3, "x", False), (0, "y", False)),
        },
    }
}


def _validate_tile_coord(ds: xr.Dataset):

    if "tile" not in ds.coords:
        raise ValueError("The input Dataset must have a `tile` coordinate.")

    if not set(ds.tile.values) == {0, 1, 2, 3, 4, 5}:
        raise ValueError("`tile` coordinate must contain each of [0, 1, 2, 3, 4, 5]")


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
            corresponding coordinate. To avoid xgcm bugs, this coordinate should start
            with 0. You can make it like this::

                ds = ds.assign_coords(tile=np.arange(6))

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
