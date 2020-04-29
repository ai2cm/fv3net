import numpy as np
import pandas as pd
import xarray as xr
import warnings

import vcm
from vcm.select import drop_nondim_coords, get_latlon_grid_coords

# give as [lat, lon]
EXAMPLE_CLIMATE_LATLON_COORDS = {
    "sahara_desert": [20.0, 10.0],
    "tropical_india": [20.0, 81.0],
    "himalayas": [28.0, 87],
    "central_canada": [55.0, 258.0],
    "tropical_west_pacific": [-5.0, 165.0],
}
_KG_M2S_TO_MM_DAY = 86400  # kg/m2/s same as mm/s. Using 1000 km/m3 for H20 density


def merge_comparison_datasets(
    data_vars, datasets, dataset_labels, concat_dim_name="dataset",
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
            with data arrays and grid.

    Returns:
        Dataset with new dataset dimension to denote the target vs predicted
        quantities. It is unstacked into the original x,y, time dimensions.
    """
    warnings.warn(
        DeprecationWarning(
            "merge_comparison_datasets is unsafe since it adds missing data variables."
        )
    )

    src_dim_index = pd.Index(dataset_labels, name=concat_dim_name)
    datasets = [drop_nondim_coords(ds) for ds in datasets]
    # if one of the datasets is missing data variable(s) that are in the others,
    # fill it with an empty data array
    _add_missing_data_vars(data_vars, datasets)
    return xr.concat(
        [ds[data_vars].squeeze(drop=True) for ds in datasets], dim=src_dim_index
    )


def get_latlon_grid_coords_set(
    grid,
    climate_latlon_coords,
    var_lat="lat",
    var_lon="lon",
    coord_x_center="x",
    coord_y_center="y",
):
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
            grid,
            latlon_coords[0],
            latlon_coords[1],
            var_lat,
            var_lon,
            coord_x_center,
            coord_y_center,
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


def net_heating_from_dataset(ds: xr.Dataset, suffix: str = None) -> xr.DataArray:
    """Compute the net heating from a dataset of diagnostic output

    This should be equivalent to the vertical integral (i.e. <>) of Q1::

        cp <Q1>

    Args:
        ds: a datasets with the names for the heat fluxes and precipitation used
            by the ML pipeline
        suffix: (optional) suffix of flux data vars if applicable. Will add '_' before
            appending to variable names if not already in suffix.

    Returns:
        the total net heating, the rate of change of the dry enthalpy <c_p T>
    """
    if suffix and suffix[0] != "_":
        suffix = "_" + suffix
    elif not suffix or suffix == "":
        suffix = ""
    fluxes = (
        ds["DLWRFsfc" + suffix],
        ds["DSWRFsfc" + suffix],
        ds["ULWRFsfc" + suffix],
        ds["ULWRFtoa" + suffix],
        ds["USWRFsfc" + suffix],
        ds["USWRFtoa" + suffix],
        ds["DSWRFtoa" + suffix],
        ds["SHTFLsfc" + suffix],
        ds["PRATEsfc" + suffix],
    )
    return vcm.net_heating(*fluxes)


def _add_empty_dataarray(ds, template_dataarray):
    """ Adds an empty data array with the dimensions of the example
    data array to the dataset. This is useful when concatenating mulitple
    datasets where one does not have a data variable.
    ex. concating prediction/target/highres datasets for
    plotting comparisons, where the high res data does not have 3D variables.

    Args:
        ds (xarray dataset): dataset that will have additional empty data array added
        example_dataarray (data array with the desired dimensions)
    
    Returns:
        original xarray dataset with an empty array assigned to the
        template name dataarray.
    
    """
    da_fill = np.empty(template_dataarray.shape)
    da_fill[:] = np.nan
    return ds.assign({template_dataarray.name: (template_dataarray.dims, da_fill)})


def _add_missing_data_vars(data_vars, datasets):
    """ Checks if any dataset in a list to be concated is missing a data variable,
    and returns of kwargs to be provided to _add_empty_dataarray
    
    Args:
        data_vars (list[str]): full list of data vars for final concated ds
        datasets ([type]): datasets to check again
    
    Returns:
        List of dicts {"ds": dataset that needs empty datarray added,
        "example_dataarray": example of data array with dims}
        This can be passed as kwargs to _add_empty_dataarray
    """
    for data_var in data_vars:
        array_var = None
        for ds in datasets:
            if data_var in list(ds.data_vars):
                array_var = ds[data_var]
        if array_var is None:
            raise ValueError(f"None of the datasets contain data array for {data_var}.")
        for i in range(len(datasets)):
            if data_var not in list(datasets[i].data_vars):
                datasets[i] = _add_empty_dataarray(datasets[i], array_var)
