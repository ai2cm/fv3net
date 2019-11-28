import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


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
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})

    MASK_SIZE = 4
    grid["grid_lont"] = grid["grid_lont"].where(
        grid.grid_lont < 180, grid.grid_lont - 360
    )
    grid["grid_lon"] = grid["grid_lon"].where(grid.grid_lon < 180, grid.grid_lon - 360)
    mask = np.abs(grid.grid_lont - 180) > MASK_SIZE
    masked = data.where(mask)

    if "vmin" not in kwargs:
        kwargs["vmin"] = float(data.min())
    if "vmax" not in kwargs:
        kwargs["vmax"] = float(data.max())
    if "cmap" not in kwargs:
        kwargs["cmap"] = "seismic"

    for tile in range(6):
        lonb = grid.grid_lon.isel(tile=tile)
        latb = grid.grid_lat.isel(tile=tile)
        im = ax.pcolormesh(
            lonb, latb, masked.isel(tile=tile), transform=ccrs.PlateCarree(), **kwargs
        )
        cf = ax.contour(
            grid.grid_lont.isel(tile=tile),
            grid.grid_latt.isel(tile=tile),
            masked.isel(tile=tile),
            transform=ccrs.PlateCarree(),
            levels=[0],
            linewidths=0.5,
        )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(data.name)

    ax.coastlines(color=[0, 0.25, 0], linewidth=1.5)

    return fig, ax
