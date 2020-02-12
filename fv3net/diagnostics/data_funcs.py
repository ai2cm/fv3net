import pandas as pd
import xarray as xr


def merge_comparison_datasets(
        var, datasets, dataset_labels, grid, additional_dataset=None):
    """ Makes a comparison dataset out of multiple datasets that all have a common
    data variable. They are concatenated with a new dim "dataset" that can be used
    to distinguish each dataset's data values from each other when plotting together.

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

    src_dim_index = pd.Index(
        dataset_labels, name="dataset"
    )
    datasets_to_merge = [xr.concat([ds[var] for ds in datasets], src_dim_index), grid]
    if additional_dataset:
        datasets += additional_dataset
    ds_comparison = xr.merge(datasets_to_merge)
    return ds_comparison

