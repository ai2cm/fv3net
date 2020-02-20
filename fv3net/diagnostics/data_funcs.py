import pandas as pd
import xarray as xr

from vcm.select import drop_nondim_coords


def merge_comparison_datasets(
    var, datasets, dataset_labels, grid, additional_dataset=None
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
        xr.concat([ds[var].squeeze(drop=True) for ds in datasets], src_dim_index),
        grid,
    ]
    if additional_dataset is not None:
        datasets_to_merge.append(additional_dataset)
    ds_comparison = xr.merge(datasets_to_merge)
    return ds_comparison
