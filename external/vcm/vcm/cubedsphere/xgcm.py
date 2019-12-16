import xarray as xr
import xgcm

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


    The tile dimension must have a corresponding coordinate. To follow GFDL's 
    convention, this coordinate should start with 1.
    
    You can make it like this::

        ds = ds.assign_coords(tile=np.arange(1, 7))


    .. _XGCM: https://xgcm.readthedocs.io/en/latest/
    .. _notebook: https://github.com/VulcanClimateModeling/explore/blob/master/noahb/2019-12-06-XGCM.ipynb # noqa

    """
    if "tile" not in ds.coords:
        raise ValueError("The input Dataset must have a `tile` coordinate.")

    coords = {
        "x": {"center": x_center, "outer": x_outer},
        "y": {"center": y_center, "outer": y_outer},
    }
    return xgcm.Grid(ds, coords=coords, face_connections=FV3_FACE_CONNECTIONS)
