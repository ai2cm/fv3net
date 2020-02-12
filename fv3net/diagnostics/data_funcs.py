import pandas as pd
import xarray as xr

from vcm.calc.thermo import LATENT_HEAT_VAPORIZATION
from vcm.select import drop_nondim_coords


def merge_comparison_datasets(
    var, datasets, dataset_labels, grid, additional_dataset=None
):
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


def energy_convergence(ds_hires):
    """

    Args:
        ds_hires: coarsened dataset created from the high res SHiELD diagnostics data

    Returns:
        Data array of the column energy convergence [W/m2]
    """
    energy_convergence = (
        (ds_hires["SHTFLsfc_coarse"] + ds_hires["LHTFLsfc_coarse"])
        + (ds_hires["USWRFsfc_coarse"] - ds_hires["USWRFtoa_coarse"])
        + (ds_hires["DSWRFtoa_coarse"] - ds_hires["DSWRFsfc_coarse"])
        + (ds_hires["ULWRFsfc_coarse"] - ds_hires["ULWRFtoa_coarse"])
        - ds_hires["DLWRFsfc_coarse"]
        + ds_hires["PRATEsfc_coarse"] * LATENT_HEAT_VAPORIZATION
    )
    return energy_convergence.rename("energy_convergence")
