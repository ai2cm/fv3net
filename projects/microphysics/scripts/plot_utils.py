import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fv3viz import plot_cube, infer_cmap_params
from fv3viz._plot_cube import _mappable_var


def meridional_ring(lon=0, n=180):
    attrs = {"description": f"Lon = {lon}"}
    lat = np.linspace(-90, 90, n)
    lon = np.ones_like(lat) * lon

    return {
        "lat": xr.DataArray(lat, dims="sample", attrs=attrs),
        "lon": xr.DataArray(lon, dims="sample", attrs=attrs),
    }


def _coords_to_points(coords, order):
    return np.stack([coords[key] for key in order], axis=-1)


def interpolate_unstructured(
    data, coords
):
    """Interpolate an unstructured dataset
    This is similar to the fancy indexing of xr.Dataset.interp, but it works
    with unstructured grids. Only nearest neighbors interpolation is supported for now.
    Args:
        data: data to interpolate
        coords: dictionary of dataarrays with single common dim, similar to the
            advanced indexing provided ``xr.DataArray.interp``. These can,
            but do not have to be actual coordinates of the Dataset, but they should
            be in a 1-to-1 map with the the dimensions of the data. For instance,
            one can use this function to find the height of an isotherm, provided
            that the temperature is monotonic with height.
    Returns:
        interpolated dataset with the coords from coords argument as coordinates.
    """
    dims_in_coords = set()
    for coord in coords:
        for dim in coords[coord].dims:
            dims_in_coords.add(dim)

    if len(dims_in_coords) != 1:
        raise ValueError(
            "The values of ``coords`` can only have one common shared "
            "dimension. The coords have these dimensions: "
            f"`{dims_in_coords}`"
        )

    dim_name = dims_in_coords.pop()

    spatial_dims = set()
    for key in coords:
        for dim in data[key].dims:
            spatial_dims.add(dim)

    stacked = data.stack({dim_name: list(spatial_dims)})
    order = list(coords)
    input_points = _coords_to_points(stacked, order)
    output_points = _coords_to_points(coords, order)
    tree = KDTree(input_points)
    _, indices = tree.query(output_points)
    output = stacked.isel({dim_name: indices})
    output = output.drop(dim_name)
    return output.assign_coords(coords)


def meridional_transect(ds: xr.Dataset):
    transect_coords = meridional_ring()
    return interpolate_unstructured(ds, transect_coords)


def plot_var(ds, name, isel_kwargs=None, plot_kwargs=None, avg_dims=None):
    if isel_kwargs is None:
        isel_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    
    selected = ds.isel(**isel_kwargs).squeeze()
    if avg_dims is not None:
        reduced = selected.mean(dim=avg_dims)
    else:
        reduced = selected
    plot_cube(reduced, name, **plot_kwargs)


def plot_meridional(ds, vkey, title="", ax=None, yincrease=False):
    ds = _mappable_var(ds, vkey)
    meridional = meridional_transect(ds)
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_dpi(120)

    if "z_soil" in meridional[vkey].dims:
        y = "z_soil"
    else:
        y = "z"

    vmin, vmax, cmap = infer_cmap_params(ds[vkey])
    meridional[vkey].plot.pcolormesh(
        x="lat", y=y, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, yincrease=yincrease,
    )
    ax.set_title(title, size=14)
    ax.set_ylabel("vertical level", size=12)
    ax.set_xlabel("latitude", size=12)
