rename = {
    "x_interface": "grid_x",
    "y_interface": "grid_y",
    "x": "grid_xt",
    "y": "grid_yt",
    "z": "pfull",
}


def gfdl_to_standard(ds):
    """Convert from GFDL dimension names (grid_xt, etc) to standard
    names (x, y, z)
    """

    key, val = rename.keys(), rename.values()
    inverse = dict(zip(val, key))

    return ds.rename({key: val for key, val in inverse.items() if key in ds.dims})


def standard_to_gfdl(ds):
    """Convert from standard names to GFDL names"""
    return ds.rename({key: val for key, val in rename.items() if key in ds.dims})
