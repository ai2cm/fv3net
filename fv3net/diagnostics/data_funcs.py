import pandas as pd
import xarray as xr

from vcm.calc.thermo import LATENT_HEAT_VAPORIZATION
from vcm.select import drop_nondim_coords, get_latlon_grid_coords

# give as [lat, lon]
EXAMPLE_CLIMATE_LATLON_COORDS = {
    "sahara_desert": [20., 10.],
    "tropical_india": [20., 81.],
    "himalayas": [28., 87],
    "central_canada": [55., 258.],
    "tropical_west_pacific": [-5., 165.]
}

def merge_comparison_datasets(
    var, datasets, dataset_labels, grid, additional_dataset=None
):
    """ Makes a comparison dataset out of multiple datasets that all have a common
    data variable. They are concatenated with a new dim "dataset" that can be used
    to distinguish each dataset's data values from each other when plotting togethe

    Args:
        var: data variable of interest that is in all datasets
        datasets: list of xr datasets or data arrays to concat
        dataset_labels: ordered list corresponding to datasets, is the coords for the
            "dataset" dimension
        grid: dataset with lat/lon grid vars
        additional_data: any additional data (e.g. slmsk) to merge along with datasets
            and grid

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """

    src_dim_index = pd.Index(dataset_labels, name="dataset")
    datasets = [drop_nondim_coords(ds) for ds in datasets]
    datasets_to_merge = [
        xr.concat([ds[var].squeeze(drop=True) for ds in datasets], src_dim_index),
        grid,
    ]
    if additional_dataset is not None:
        datasets_to_merge.append(additional_dataset)
    ds_comparison = xr.merge(datasets_to_merge)
    return ds_comparison


def hires_diag_column_heating(ds_hires):
    """

    Args:
        ds_hires: coarsened dataset created from the high res SHiELD diagnostics data

    Returns:
        Data array of the column energy convergence [W/m2]
    """
    heating = (
        ds_hires["SHTFLsfc_coarse"]
        + (ds_hires["USWRFsfc_coarse"] - ds_hires["USWRFtoa_coarse"])
        + (ds_hires["DSWRFtoa_coarse"] - ds_hires["DSWRFsfc_coarse"])
        + (ds_hires["ULWRFsfc_coarse"] - ds_hires["ULWRFtoa_coarse"])
        - ds_hires["DLWRFsfc_coarse"]
        + ds_hires["PRATEsfc_coarse"] * LATENT_HEAT_VAPORIZATION
    )
    return heating.rename("heating")


def get_latlon_grid_coords(grid, climate_latlon_coords):
    climate_grid_coords = {}
    for climate, latlon_coords in climate_latlon_coords.items():
        climate_grid_coords[climate] = get_latlon_grid_coords(grid, lat=latlon[0], lon=latlon[1])
    return climate_grid_coords