"""
Some helper function for visualization.
"""
import holoviews as hv


def make_image(sliced_data, cmap_range=None, **kwargs):
    coords = ['grid_xt', 'grid_yt'] if 'grid_xt' in sliced_data.coords else ['lon', 'lat']
    hv_img = hv.Image(sliced_data, coords)
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(**{var_name: hv.Dimension(var_name, range=cmap_range)})
    hv_img = hv_img.options(**kwargs)
    return hv_img

def make_animation(sliced_data, cmap_range=None, **kwargs):
    coords = ['grid_xt', 'grid_yt'] if 'grid_xt' in sliced_data.coords else ['lon', 'lat']
    hv_ds = hv.Dataset(sliced_data)
    hv_img = hv_ds.to(hv.Image, coords).options(**kwargs)
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(**{var_name: hv.Dimension(var_name, range=cmap_range)})
    return hv_img