import numpy as np
import vcm.convenience

rename = {
    "x_interface": "grid_x",
    "y_interface": "grid_y",
    "x": "grid_xt",
    "y": "grid_yt",
    "z": "pfull",
}


def gfdl_to_standard(ds):

    key, val = rename.keys(), rename.values()
    inverse = dict(zip(val, key))

    return ds.rename({key: val for key, val in inverse.items() if key in ds.dims})


def standard_to_gfdl(ds):
    return ds.rename({key: val for key, val in rename.items() if key in ds.dims})


def round_time(ds):
    times = np.vectorize(vcm.convenience.round_time)(ds.time)
    return ds.assign_coords(time=times)
