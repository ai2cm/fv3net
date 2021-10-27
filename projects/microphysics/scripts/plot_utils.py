import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fv3viz._plot_cube import plot_cube, _mappable_var
from vcm.catalog import catalog

GRID = catalog["grid/c48"].to_dask()

MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}


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


def to_mappable(da, name):
    merged_w_grid = da.to_dataset(name=name).merge(GRID, join="outer")
    return _mappable_var(merged_w_grid, name)


def plot_var(ds, name, isel_kwargs=None, plot_kwargs=None, avg_dims=None):
    if isel_kwargs is None:
        isel_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    
    mappable = to_mappable(ds[name].transpose(..., "tile"), name)
    selected = mappable.isel(**isel_kwargs).squeeze()
    if avg_dims is not None:
        reduced = selected.mean(dim=avg_dims)
    else:
        reduced = selected
    plot_cube(reduced, name, **plot_kwargs)


def plot_meridional(da, vkey, title="", ax=None, vmin=None, vmax=None, yincrease=False):
    merged = da.to_dataset(name=vkey).merge(GRID)
    meridional = meridional_transect(_mappable_var(merged, vkey))
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_dpi(120)
    if "z_soil" in meridional.dims:
        y = "z_soil"
    else:
        y = "z"
    meridional[vkey].plot.pcolormesh(
        x="lat", y=y, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax, yincrease=yincrease,
#         cbar_kwargs={"label": meta[vkey]["units"]},
    )
    ax.set_title(title, size=14)
    ax.set_ylabel("vertical level", size=12)
    ax.set_xlabel("latitude", size=12)
