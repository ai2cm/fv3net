import matplotlib.pyplot as plt
import numpy as np

# map plotting function waiting on another PR to get merged
# from vcm.visualize import plot_cube

PLOT_TYPES = ['map', 'time_series']

VERTICAL_GRID_VAR = 'pfull'
LON_GRID_CENTER = 'grid_lont'
LAT_GRID_CENTER = 'grid_latt'
LON_GRID_EDGE = 'grid_lon'
LAT_GRID_EDGE = 'grid_lat'


def create_plot(
        ds,
        plot_config
):
    if plot_config.plot_type == 'map':
        return _plot_var_map(ds, plot_config)
    elif plot_config.plot_type == 'time_series':
        return _plot_var_time_series(ds, plot_config)
    else:
        raise ValueError("Invalid plot_type in config, must be in {}".format(PLOT_TYPES))


def _get_grid(ds):
    return ds[[LAT_GRID_EDGE, LON_GRID_EDGE, LAT_GRID_CENTER, LON_GRID_CENTER]]


def _plot_var_map(
        ds,
        plot_config,
):
    """

    Args:
        ds: xr dataset
        plot_config: dict that specifies variable and dimensions to plot,
        functions to apply
        plot_func: optional plotting function from vcm.visualize
    Returns:
        axes
    """
    # sacrificing some ugliness here so that specification in the yaml is safer / more readable:
    # single entry dicts seemed easier to write in yaml than list of func name and kwargs
    for dim, dim_slice in plot_config.dim_slices.items():
        ds = ds.isel({dim: dim_slice})
    for function, kwargs in zip(plot_config.functions, plot_config.function_kwargs):
        ds = ds.pipe(function, **kwargs)
    grid = _get_grid(ds)
    fig, ax = plot_cube(ds[plot_config.diagnostic_variable], grid)
    return fig


def _plot_var_time_series(ds, plot_config):
    pass


def _plot_histogram(ds, plot_config):
    pass



# temporary while waiting on PR! will remove after #67 is merged.
import xarray as xr
import numpy as np
import cartopy.crs as ccrs


def plot_cube(data: xr.DataArray, grid: xr.Dataset, ax=None, colorbar=True, **kwargs):
    """ Plots cubed sphere grids into a global map projection

    Arguments:

    Data: Dataarray of variable to plot assumed to have dimensions (grid_ty, grid_xt and tile)

    grid: Dataset of grid variables that must include:
        -lat, lon, latb, lonb (where lat and lon are grid centers and lonb and latb are grid edges)

    Returns:

    Fig and ax handles

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()})

    MASK_SIZE = 4
    grid['grid_lont'] = grid['grid_lont'].where(grid.grid_lont < 180, grid.grid_lont - 360)
    grid['grid_lon'] = grid['grid_lon'].where(grid.grid_lon < 180, grid.grid_lon - 360)
    mask = np.abs(grid.grid_lont - 180) > MASK_SIZE
    masked = data.where(mask)

    if 'vmin' not in kwargs:
        kwargs['vmin'] = float(data.min())
    if 'vmax' not in kwargs:
        kwargs['vmax'] = float(data.max())
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'seismic'

    for tile in range(6):
        lonb = grid.grid_lon.isel(tile=tile)
        latb = grid.grid_lat.isel(tile=tile)
        im = ax.pcolormesh(
            lonb,
            latb,
            masked.isel(tile=tile),
            transform=ccrs.PlateCarree(),
            **kwargs)
        cf = ax.contour(
            grid.grid_lont.isel(tile=tile),
            grid.grid_latt.isel(tile=tile),
            masked.isel(tile=tile),
            transform=ccrs.PlateCarree(),
            levels=[0],
            linewidths=0.5
        )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(data.name)

    ax.coastlines(color=[0, 0.25, 0], linewidth=1.5)

    return fig, ax