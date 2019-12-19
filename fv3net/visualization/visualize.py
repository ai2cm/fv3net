"""
Some helper function for visualization.
"""
import holoviews as hv
from vcm.cubedsphere.constants import COORD_X_CENTER, COORD_Y_CENTER


def make_image(
    sliced_data, cmap_range=None, coords=None, quad=False, invert_y=False, **kwargs
):
    coords = coords or (
        [COORD_X_CENTER, COORD_Y_CENTER]
        if COORD_X_CENTER in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_img = hv.QuadMesh(sliced_data, coords) if quad else hv.Image(sliced_data, coords)
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(**{var_name: hv.Dimension(var_name, range=cmap_range)})
    hv_img = hv_img.options(**kwargs)
    return hv_img


def make_animation(
    sliced_data, cmap_range=None, coords=None, quad=False, invert_y=False, **kwargs
):
    coords = coords or (
        [COORD_X_CENTER, COORD_Y_CENTER]
        if COORD_X_CENTER in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_ds = hv.Dataset(sliced_data)
    hv_img = hv_ds.to(hv.QuadMesh if quad else hv.Image, coords).options(**kwargs)
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(**{var_name: hv.Dimension(var_name, range=cmap_range)})
    return hv_img
