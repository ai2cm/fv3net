from vcm.cubedsphere.constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
)
from vcm.calc.transform_cubesphere_coords import get_rotated_centered_winds
from vcm.visualize.plot_helpers import _infer_color_limits, _get_var_label
from vcm.visualize.masking import _mask_antimeridian_quads
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
import warnings
from functools import partial
from typing import Callable


# globals

RESTART_COORD_VARS = {
    "grid_lon": [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    "grid_lat": [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    "grid_lont": [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
    "grid_latt": [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
}

DIAG_COORD_VARS = {
    "lonb": [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    "latb": [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    "lon": [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
    "lat": [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
}

COORD_NAME_MAPPING = {
    "grid_lon": "lonb",
    "grid_lat": "latb",
    "grid_lont": "lon",
    "grid_latt": "lat",
}


def plot_cube(
    plottable_variable: xr.Dataset,
    plotting_function: Callable = plt.pcolormesh,
    ax: plt.axes = None,
    row: str = None,
    col: str = None,
    col_wrap: int = None,
    projection: ccrs.Projection = ccrs.Robinson(),
    colorbar: bool = True,
    coastlines: bool = True,
    coastlines_kwargs: dict = None,
    **kwargs,
):
    """ Plots tiled cubed sphere grids onto a global map projection
    
    Args:
    
        plottable_variable (xr.Dataset):
            Dataset containing variable to plotted via pcolormesh, along with
            coordinate variables (lat, latb, lon, lonb). This dataset object 
            can be created from the helper function `mappable_restart_var`,
             which takes in the output of `vcm.v3_restarts.open_restarts` 
            merged to  dataset of grid spec variables, along with the name of 
            the variable to be plotted, or from the helper function 
            `mappable_diag_var`, which takes in the output of 
            `vcm.v3_restarts.open_restarts`
        plotting_function (Callable, optional):
            Matplotlib 2-d plotting function to use in plotting the variable. 
            Available options are plt.pcolormesh,  plt.contour, and
            plt.contourf. Defaults to plt.pcolormesh.
        ax (plt.axes, optional):
            Axes onto which the map should be plotted; must be created with 
            a cartopy projection argument. If not supplied, axes are generated 
            with a projection. If ax is suppled, faceting is disabled. 
        row (str, optional): 
            Name of diemnsion to be faceted along subplot rows. Must not be a 
            tile, lat, or lon dimension.  Defaults to no row facets.
        col (str, optional): 
            Name of diemnsion to be faceted along subplot columns. Must not be 
            a tile, lat, or lon dimension. Defaults to no column facets.
        col_wrap (int, optional): 
            If only one of `col`, `row` is specified, number of columns to plot
            before wrapping onto next row. Defaults to None, i.e. no limit.
        projection (ccrs.Projection, optional):
            Cartopy projection object to be used in creating axes. Ignored if 
            cartopy geo-axes are supplied.  Defaults to Robinson projection. 
        colorbar (bool, optional):
            Flag for whether to plot a colorbar. Defaults to True.
        coastlines (bool, optinal):
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs (dict, optional):
            Dict of arguments to be passed to cartopy axes's `coastline`
            function if `coastlines` flag is set to True.
        **kwargs:
            Additional keyword arguments to be passed to the plotting function.
    
    Returns:
    
        axes (list):
            List or nested list of `plt.axes` objects assocated with map 
            subplots if faceting; otherwise single `plt.axes` object.
        handles (list):
            List or nested list of matplotlib object handles associated with 
            map subplots if faceting; otherwise single object handle.
        cbar (obj):
            `ax.colorbar` object handle associated with figure, if `colorbar` 
            arg is True, else None. 
    
    Example:
    
        # plots T at multiple vertical levels, faceted across subplots
        sample_data = fv3_restarts.open_restarts(
                '/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/
                rundir/',
                '20160805.170000',
                '20160805.171500'
            )
        grid_spec_paths = [f"/home/brianh/dev/fv3net/data/restart/C48/
        20160805.170000/rundir/grid_spec.tile{tile}.nc" for tile in range (1,7)]
        grid_spec = xr.open_mfdataset(paths = grid_spec_paths, 
        combine = 'nested', concat_dim = 'tile')
        ds = xr.merge([sample_data, grid_spec])
        axes, hs, cbar = plot_cube(
            mappable_restart_var(ds, 'T').isel(time = 0, pfull = [78, 40]),
            plotting_function='pcolormesh',
            row = "pfull",
            coastlines = True,
            coastlines_kwargs = coastlines_kwargs,
            colorbar = True,
            vmin = 250,
            vmax = 300,
        )
    """

    var_name = list(plottable_variable.data_vars)[0]
    array = plottable_variable[var_name].values

    xmin = np.nanmin(array)
    xmax = np.nanmax(array)
    vmin = kwargs["vmin"] if "vmin" in kwargs else None
    vmax = kwargs["vmax"] if "vmax" in kwargs else None
    cmap = kwargs["cmap"] if "cmap" in kwargs else None
    kwargs["vmin"], kwargs["vmax"], kwargs["cmap"] = _infer_color_limits(
        xmin, xmax, vmin, vmax, cmap
    )

    _plot_func_short = partial(
        _plot_cube_axes,
        lat=plottable_variable.lat.values,
        lon=plottable_variable.lon.values,
        latb=plottable_variable.latb.values,
        lonb=plottable_variable.lonb.values,
        plotting_function=plotting_function,
        coastlines=coastlines,
        coastlines_kwargs=coastlines_kwargs,
        **kwargs,
    )

    if not ax and (row or col):
        # facets
        facet_grid = xr.plot.FacetGrid(
            data=plottable_variable,
            row=row,
            col=col,
            col_wrap=col_wrap,
            subplot_kws={"projection": projection},
        )
        facet_grid = facet_grid.map(_plot_func_short, var_name)
        axes = facet_grid.axes
        handles = facet_grid._mappables
    else:
        # single axes
        if not ax:
            f, ax = plt.subplots(1, 1, subplot_kw={"projection": projection})
        handle = _plot_func_short(array)
        axes = [ax]
        handles = [handle]

    if colorbar:
        plt.gcf().subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
        )
        cb_ax = plt.gcf().add_axes([0.83, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(handles[0], cax=cb_ax, extend="both")
        cbar.set_label(_get_var_label(plottable_variable[var_name].attrs, var_name))
    else:
        cbar = None

    return axes, handles, cbar


def mappable_var(ds: xr.Dataset, var_name: str, ds_type: str = "restart"):

    """ Converts a restart or diagnostic dataset into a format for plotting
    across cubed-sphere tiles
    
    Args:
    
        ds (xr.Dataset): 
            Dataset containing the variable to be plotted, along with grid spec 
            information. Assumed to be created by merging 
            `fv3_restarts.open_restarts` output and grid spec tiles, or 
            `fv3_restarts.open_standard_diags`.
        var_name (str): 
            Name of variable to be plotted.
        ds_type (str, optional):
            A string indicating the type of dataset ('restart' or 'diag') 
            from which the variable is being plotted. Defaults to 'restart'.
    
    Returns:
    
        ds (xr.Dataset):
            Dataset containing variable to be plotted as well as grid 
            coordinates variables. Grid variables are renamed and ordered for 
            plotting as first argument to `plot_cube`.
    
    Example:
    
        # plots T at multiple vertical levels, faceted across subplots
        sample_data = fv3_restarts.open_restarts(
                '/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/
                rundir/',
                '20160805.170000',
                '20160805.171500'
            )
        grid_spec_paths = [f"/home/brianh/dev/fv3net/data/restart/C48/
        20160805.170000/rundir/grid_spec.tile{tile}.nc" for tile in range (1,7)]
        grid_spec = xr.open_mfdataset(paths = grid_spec_paths, 
        combine = 'nested', concat_dim = 'tile')
        ds = xr.merge([sample_data, grid_spec])
        axes, hs, cbar = plot_cube(
            mappable_restart_var(ds, 'T').isel(time = 0, pfull = [78, 40]),
            plotting_function='pcolormesh',
            row = "pfull",
            coastlines = True,
            coastlines_kwargs = coastlines_kwargs,
            colorbar = True,
            vmin = 250,
            vmax = 300,
        )
    """

    if ds_type not in ["restart", "diag"]:
        raise ValueError('ds_type must be "restart" or "diag".')

    if ds_type == "restart":
        coord_dict = RESTART_COORD_VARS
    else:
        coord_dict = DIAG_COORD_VARS

    for var, dims in coord_dict.items():
        ds[var] = ds[var].transpose(*dims)

    if var_name in ["u", "v"] and ds_type == "restart":

        u_r, v_r = get_rotated_centered_winds(ds)

        if var_name == "u":
            ds = ds.assign({var_name: u_r})
        else:
            ds = ds.assign({var_name: v_r})

    new_ds = (
        ds[[var_name]].copy().transpose(COORD_Y_CENTER, COORD_X_CENTER, "tile", ...)
    )

    for restart_name, diag_name in COORD_NAME_MAPPING.items():
        if ds_type == "restart":
            new_ds = new_ds.assign_coords(coords={diag_name: ds[restart_name]})
        else:
            new_ds = new_ds.assign_coords(coords={diag_name: ds[diag_name]})

    return new_ds.drop(
        labels=[COORD_Y_CENTER, COORD_X_CENTER, COORD_Y_OUTER, COORD_X_OUTER]
    )


def _plot_cube_axes(
    array: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    plotting_function: str,
    coastlines: bool = False,
    coastlines_kwargs: dict = None,
    title: str = None,
    **kwargs,
):

    """ Plots tiled cubed sphere for a given subplot axis,
        using np.ndarrays for all data
    
    Args:
    
        array (np.ndarray): 
            Array of variables values at cell centers, of dimensions (npy, npx,
            tile)
        lat (np.ndarray): 
            Array of latitudes of cell centers, of dimensions (npy, npx, tile) 
        lon (np.ndarray): 
            Array of longitudes of cell centers, of dimensions (npy, npx, tile)
        latb (np.ndarray): 
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile) 
        lonb (np.ndarray): 
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile) 
        coastlines (bool, optinal):
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs (dict, optional):
            Dict of options to be passed to cartopy axes's `coastline` function
            if `coastlines` flag is set to True.
        title (str, optional):
            Title text of subplot. Defaults to None.
        **kwargs:
            Keyword arguments to be passed to plotting function. 
    
    Returns:
    
        p_handle (obj):
            matplotlib object handle associated with map subplot
    
    """

    if (lon.shape[-1] != 6) or (lat.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError(
            """Last axis of each array must have six elements for 
            cubed-sphere tiles."""
        )

    if (
        (lon.shape[0] != lat.shape[0])
        or (lat.shape[0] != array.shape[0])
        or (lon.shape[1] != lat.shape[1])
        or (lat.shape[1] != array.shape[1])
    ):
        raise ValueError(
            """First and second axes lengths of lat and lonb must be equal to 
            those of array."""
        )

    if (len(lonb.shape) != 3) or (len(latb.shape) != 3) or (len(array.shape) != 3):
        raise ValueError("Lonb, latb, and data_var each must be 3-dimensional.")

    if (lonb.shape[-1] != 6) or (latb.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError(
            "Last axis of each array must have six elements for cubed-sphere tiles."
        )

    if (
        (lonb.shape[0] != latb.shape[0])
        or (latb.shape[0] != (array.shape[0] + 1))
        or (lonb.shape[1] != latb.shape[1])
        or (latb.shape[1] != (array.shape[1] + 1))
    ):
        raise ValueError(
            """First and second axes lengths of latb and lonb 
            must be one greater than those of array."""
        )

    if (len(lon.shape) != 3) or (len(lat.shape) != 3) or (len(array.shape) != 3):
        raise ValueError("Lonb, latb, and data_var each must be 3-dimensional.")

    ax = plt.gca()

    if plotting_function in [plt.pcolormesh, plt.contour, plt.contourf]:
        setattr(ax, "plotting_function", plotting_function)
    else:
        raise ValueError(
            """Plotting functions only include plt.pcolormesh, plt.contour, 
            and plt.contourf."""
        )

    central_longitude = ax.projection.proj4_params["lon_0"]

    masked_array = np.where(
        _mask_antimeridian_quads(lonb, central_longitude), array, np.nan
    )

    for tile in range(6):
        if ax.plotting_function == plt.pcolormesh:
            x = lonb[:, :, tile]
            y = latb[:, :, tile]
        else:
            # contouring
            lon_tile = lon[:, :, tile]
            x = np.where(
                lon_tile < (central_longitude + 180.0) % 360.0,
                lon_tile,
                lon_tile - 360.0,
            )
            y = lat[:, :, tile]
            if "levels" not in kwargs:
                kwargs["n_levels"] = (
                    11 if "n_levels" not in kwargs else kwargs["n_levels"]
                )
                kwargs["levels"] = np.linspace(
                    kwargs["vmin"], kwargs["vmax"], kwargs["n_levels"]
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_handle = ax.plotting_function(
                x, y, masked_array[:, :, tile], transform=ccrs.PlateCarree(), **kwargs
            )

    ax.set_global()

    if coastlines:
        coastlines_kwargs = dict() if not coastlines_kwargs else coastlines_kwargs
        ax.coastlines(**coastlines_kwargs)

    return p_handle
