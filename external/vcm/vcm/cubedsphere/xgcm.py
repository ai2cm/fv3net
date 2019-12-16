import xarray as xr
import xgcm

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

# diagnostics defaults
COORD_X_CENTER = "grid_xt"
COORD_X_OUTER = "grid_x"
COORD_Y_CENTER = "grid_yt"
COORD_Y_OUTER = "grid_y"


def create_fv3_grid(
    ds: xr.Dataset,
    x_center: str = COORD_X_CENTER,
    x_outer: str = COORD_X_OUTER,
    y_center: str = COORD_Y_CENTER,
    y_outer: str = COORD_Y_OUTER,
) -> xgcm.Grid:
    """Create an XGCM_ grid from a dataset of FV3 tile data

    This object can be used to interpolate and differentiate cubed sphere data, please
    see the XGCM_ documentation for more information.

    See this notebook_ for usage.


    The tile dimension must have a corresponding coordinate. You can make it like this::

        ds = ds.assign_coords(tile=np.arange(6))


    .. _XGCM: https://xgcm.readthedocs.io/en/latest/
    .. _notebook: https://github.com/VulcanClimateModeling/explore/blob/master/noahb/2019-12-06-XGCM.ipynb # noqa

    """
    if "tile" not in ds.coords:
        raise ValueError("The input Dataset must have a `tile` coordinate.")

    coords = {
        "x": {"center": COORD_X_CENTER, "outer": COORD_X_OUTER},
        "y": {"center": COORD_Y_CENTER, "outer": COORD_Y_OUTER},
    }
    return xgcm.Grid(ds, coords=coords, face_connections=FV3_FACE_CONNECTIONS)
