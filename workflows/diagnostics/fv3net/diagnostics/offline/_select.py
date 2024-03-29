import xarray as xr
import cftime

from vcm.select import meridional_ring
import vcm


def select_snapshot(ds: xr.Dataset, time: cftime.DatetimeJulian) -> xr.Dataset:
    return ds.sortby("time").sel(time=time, method="nearest")


def meridional_transect(ds: xr.Dataset):
    transect_coords = meridional_ring()
    return vcm.interpolate_unstructured(ds, transect_coords)


def plot_transect(
    data: xr.DataArray,
    xaxis: str = "lat",
    yaxis: str = "pressure",
    column_dim: str = "derivation",
    dataset_dim: str = "dataset",
):
    row_dim = dataset_dim if dataset_dim in data.dims else None
    num_datasets = len(data[dataset_dim]) if dataset_dim in data.dims else 1
    figsize = (10, 4 * num_datasets)
    facetgrid = data.plot(
        y=yaxis,
        x=xaxis,
        yincrease=False,
        col=column_dim,
        row=row_dim,
        figsize=figsize,
        robust=True,
    )
    facetgrid.set_ylabels("Pressure [Pa]")
    facetgrid.set_xlabels("Latitude [deg]")

    f = facetgrid.fig
    return f
