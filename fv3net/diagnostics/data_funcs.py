import numpy as np
import pandas as pd
import xarray as xr

from vcm.select import drop_nondim_coords, get_latlon_grid_coords

# give as [lat, lon]
EXAMPLE_CLIMATE_LATLON_COORDS = {
    "sahara_desert": [20.0, 10.0],
    "tropical_india": [20.0, 81.0],
    "himalayas": [28.0, 87],
    "central_canada": [55.0, 258.0],
    "tropical_west_pacific": [-5.0, 165.0],
}


def save_lines(ax, title, saved_data=None):
    if not saved_data:
        saved_data = {}
    for line in ax.lines:
        x, y, label = line.get_xdata(), line.get_ydata(), line.get_label()
        saved_data[title] = {"x": x, "y": y, "label": label}
    return saved_data


def merge_comparison_datasets(
    data_vars, datasets, dataset_labels, grid, additional_dataset=None
):
    """ Makes a comparison dataset out of multiple datasets that all have a common
    data variable. They are concatenated with a new dim "dataset" that can be used
    to distinguish each dataset's data values from each other when plotting together.

    Args:
        var: str, data variable of interest that is in all datasets
        datasets: list[xr datasets or data arrays], arrays that will be concatenated
            along the dataset dimension
        dataset_labels: list[str], same order that corresponds to datasets,
            is the coords for the "dataset" dimension
        grid: xr dataset with lat/lon grid vars
        additional_data: xr data array, any additional data (e.g. slmsk) to merge along
            with data arrays and grid

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """

    src_dim_index = pd.Index(dataset_labels, name="dataset")
    datasets = [drop_nondim_coords(ds) for ds in datasets]
    datasets_to_merge = [
        xr.concat([ds[data_vars].squeeze(drop=True) for ds in datasets], src_dim_index),
        grid,
    ]
    if additional_dataset is not None:
        datasets_to_merge.append(additional_dataset)
    ds_comparison = xr.merge(datasets_to_merge)
    return ds_comparison


def get_latlon_grid_coords_set(grid, climate_latlon_coords):
    """ Create a dict of {location: ds.sel dict args} out of an
    input dict of {location: [lat, lon]}. This is useful for
    showing data in a fixed location.
    
    Args:
        grid (xr dataset): has lat, lon as data variables
        climate_latlon_coords (dict): has format {location: [lat, lon]}
    
    Returns:
        dict: {location: ds.sel dict args},
            e.g. {
                "indonesia": {"tile": 2, "grid_xt": 3, "grid_yt": 23},
                "vancouver": {"tile": 2, "grid_xt": 3, "grid_yt": 23},
            }
    """
    climate_grid_coords = {}
    for climate, latlon_coords in climate_latlon_coords.items():
        climate_grid_coords[climate] = get_latlon_grid_coords(
            grid, lat=latlon_coords[0], lon=latlon_coords[1]
        )
    return climate_grid_coords


def periodic_phase(phase):
    """normalize phases to be in [0, 360] deg
    
    Args:
        phase (array): phases in degrees

    Returns:
        [array]: normalized phases
    """

    def _conditions(d):
        if d > 0:
            return d - int(d / 360) * 360
        else:
            return d - int((d / 360) - 1) * 360

    cond = np.vectorize(_conditions)
    return cond(phase)
