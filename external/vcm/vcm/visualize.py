from vcm import fv3_restarts, cubedsphere
from vcm.calc.transform_cubesphere_coords import mask_antimeridian_quads
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
from os.path import join
import warnings
from functools import partial


# globals

RESTART_COORD_VARS = {'grid_lon' : ['grid_y', 'grid_x', 'tile'],
              'grid_lat' : ['grid_y', 'grid_x', 'tile'],
              'grid_lont' : ['grid_yt', 'grid_xt', 'tile'],
              'grid_latt' : ['grid_yt', 'grid_xt', 'tile']
             }

DIAG_COORD_VARS = {'lonb' : ['grid_y', 'grid_x', 'tile'],
              'latb' : ['grid_y', 'grid_x', 'tile'],
              'lon' : ['grid_yt', 'grid_xt', 'tile'],
              'lat' : ['grid_yt', 'grid_xt', 'tile']
             }


def plot_cube(
    plottable_variable: xr.Dataset,
    plotting_function: str = 'pcolormesh',
    ax: plt.axes = None,
    row: str = None,
    column: str = None,
    projection: ccrs.Projection = ccrs.Robinson(),
    colorbar: bool = True,
    coastlines: bool = True,
    coastlines_kwargs: dict = None,
    **kwargs
):
    """ Plots tiled cubed sphere grids onto a global map projection
    
    Args:
    
        plottable_variable (xr.Dataset): 
            Dataset containing variable to plotted via pcolormesh, along with 
            coordinate variables (lat, latb, lon, lonb). This dataset object 
            can be created from the helper function `mappable_restart_var`, which
            takes in the output of `vcm.v3_restarts.open_restarts` merged to 
            dataset of grid spec variables, along with the name of the variable to be plotted, or from the helper function `mappable_diag_var`, 
            which takes in the output of `vcm.v3_restarts.open_restarts`
        plotting_function (str):
            Function name to use in plotting the variable. Available options are 'pcolormesh' and 'contour'. Defaults to 'pcolormesh'.
        ax (plt.axes, optional):
            Axes onto which the map should be plotted; must be created with a cartopy projection argument. 
            If not supplied, axes are generated with a projection. If axes are suppled, faceting is disabled
            and the `row` and `column` arguments are ignored. 
        row (str, optional): 
            Name of diemnsion to be faceted along subplot rows. Must not be a tile, lat, or lon dimension. 
            Defaults to no row facets.
        column (str, optional): 
            Name of diemnsion to be faceted along subplot columns. Must not be a tile, lat, or lon dimension.
            Defaults to no column facets.
        projection (ccrs.Projection, optional):
            Cartopy projection object to be used in creating axes. Ignored if cartopy geo-axes are supplied. 
            Defaults to Robinson projection. 
        colorbar (bool, optional):
            Flag for whether to plot a colorbar. Defaults to True.
        coastlines (bool, optinal):
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs (dict, optional):
            Dict of options to be passed to cartopy axes's `coastline` function if `coastlines` flag is set to True.
    
    Returns:
    
        axes (list):
            List or nested list of `ax.axes` objects assocated with map subplots.
        hs (list):
            List or nested list of matplotlib object handles associated with map subplots.
        cbar (obj):
            `ax.colorbar` object handle associated with figure
    
    Example:
    
        # plots T at multiple vertical levels, faceted across subplots
        sample_data = fv3_restarts.open_restarts(
                '/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/rundir/',
                '20160805.170000',
                '20160805.171500'
            )
        grid_spec_paths = [f"/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/rundir/grid_spec.tile{tile}.nc" for tile in range (1,7)]
        grid_spec = xr.open_mfdataset(paths = grid_spec_paths, combine = 'nested', concat_dim = 'tile')
        ds = xr.merge([sample_data, grid_spec])
        axes, hs, cbar = plot_cube(
            mappable_restart_var(ds, 'T').isel(time = 0, pfull = [78, 40]),#.isel(pfull = slice(None, None, 20)),
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
    vmin = kwargs['vmin'] if 'vmin' in kwargs else None
    vmax = kwargs['vmax'] if 'vmax' in kwargs else None
    cmap = kwargs['cmap'] if 'cmap' in kwargs else None
    kwargs["vmin"], kwargs["vmax"], kwargs["cmap"] = _infer_color_limits(xmin, xmax, vmin, vmax, cmap)
    
    _plot_func_short = partial(
        _plot_cube_axes,
        plotting_function,
        plottable_variable.lat.values,
        plottable_variable.lon.values,
        plottable_variable.latb.values,
        plottable_variable.lonb.values,
        coastlines = coastlines,
        coastlines_kwargs = coastlines_kwargs,
        **kwargs
    )
    
    if not ax and (row or column):
        # facets

        if row and row not in plottable_variable.dims:
            raise ValueError('Row not in dataset dimensions.')
        if column and column not in plottable_variable.dims:
            raise ValueError('Column not in dataset dimensions.')
        remaining_dims = set([dim for dim in plottable_variable.dims if dim not in [row, column]])
        if remaining_dims != set(['grid_x', 'grid_xt', 'grid_y', 'grid_yt', 'tile']):
            raise valueError('Dimensions for each facet plot must consist only of latitude, longitude, and tile.')
        n_rows = plottable_variable.sizes[row] if row else 1
        n_cols = plottable_variable.sizes[column] if column else 1
        if n_rows > 10 or n_cols > 10:
            raise ValueError('Facet rows and/or columns exceed maximum. Try subsetting along the row and/or column dimensions.')
            
        _, axes = plt.subplots(n_rows, n_cols, subplot_kw={'projection': projection})
        
        hs = []
        for i in range(axes.shape[0]):
            if len(axes.shape) > 1:
                hs2 = []
                for j in range(axes.shape[1]):
                    h = _plot_func_short(
                        array = plottable_variable[var_name].isel({
                            row : i,
                            column : j
                        }).values,
                        ax = axes[i, j],
                        title = f"{row} = {plottable_variable[row].isel({row : i}).item()}, {column} = {plottable_variable[column].isel({column : j}).item()}"
                    )
                    hs2.append(h)
                hs.append(hs2)
            else:
                coord = row if not column else column
                h = _plot_func_short(
                    array = plottable_variable[var_name].isel({coord : i}).values,
                    ax = axes[i],
                    title = f"{coord} = {plottable_variable[coord].isel({coord : i}).item()}"
                    )
                hs.append(h)
    else:
        # single axes
        
        if not ax:
            _, ax = plt.subplots(1, 1, subplot_kw={'projection': projection})
        h = _plot_func_short(
            array = array,
            ax = ax
        )
        axes = [ax]
        hs = [h]
        
    if colorbar:
        plt.gcf().subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)
        cb_ax = plt.gcf().add_axes([0.83, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(h, cax=cb_ax)
        cbar.ax.set_ylabel(var_name)
    else:
        cbar = None
    
    return axes, hs, cbar


def mappable_restart_var(
    restart_ds: xr.Dataset,
    var_name: str
):
    
    """ Converts a restart dataset into a format for plotting across cubed-sphere tiles
    
    Args:
    
        restart_ds (xr.Dataset): 
            Dataset containing the variable to be plotted, along with grid spec information. Assumed to be
            created by merging `fv3_restarts.open_restarts` output and grid spec tiles. 
        var_name (str): 
            Name of variable to be plotted.
    
    Returns:
    
        ds (xr.Dataset):
            Dataset containing variable to be plotted as well as grid coordinates variables.
            Grid variables are renamed and ordered for plotting as first argument to `plot_cube`.
    
    Example:
    
        # plots T at multiple vertical levels, faceted across subplots
        sample_data = fv3_restarts.open_restarts(
                '/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/rundir/',
                '20160805.170000',
                '20160805.171500'
            )
        grid_spec_paths = [f"/home/brianh/dev/fv3net/data/restart/C48/20160805.170000/rundir/grid_spec.tile{tile}.nc" for tile in range (1,7)]
        grid_spec = xr.open_mfdataset(paths = grid_spec_paths, combine = 'nested', concat_dim = 'tile')
        ds = xr.merge([sample_data, grid_spec])
        axes, hs, cbar = plot_cube(
            mappable_restart_var(ds, 'T').isel(time = 0, pfull = [78, 40]),#.isel(pfull = slice(None, None, 20)),
            plotting_function='pcolormesh',
            row = "pfull",
            coastlines = True,
            coastlines_kwargs = coastlines_kwargs,
            colorbar = True,
            vmin = 250,
            vmax = 300,
        )
    """
    
    for var, dims in RESTART_COORD_VARS.items():
        restart_ds[var] = restart_ds[var].transpose(*dims)
        
    if var_name in ['u', 'v']:
        u_c = cubedsphere.shift_edge_var_to_center(restart_ds['u'].drop(labels = 'grid_xt'))
        v_c = cubedsphere.shift_edge_var_to_center(restart_ds['v'].drop(labels = 'grid_yt'))
        u_r, v_r = rotate_winds_to_lat_lon_coords(u_c, v_c,
            restart_ds[['grid_lont', 'grid_latt', 'grid_lon', 'grid_lat']]
            )
        if var_name == 'u':
            restart_ds = restart_ds.assign({var_name: u_r})
        else:
            restart_ds = restart_ds.assign({var_name: v_r})
    
    return (
        restart_ds[[var_name]]
        .transpose('grid_yt', 'grid_xt', 'tile', 'pfull', 'time')
        .assign_coords(
            coords = {
                "grid_lont" : restart_ds["grid_lont"],
                "grid_latt" : restart_ds["grid_latt"],
                "grid_lon" : restart_ds['grid_lon'],
                "grid_lat" : restart_ds['grid_lat']
            }
        )
        .drop(labels = ['grid_xt', 'grid_yt', 'grid_x', 'grid_y', 'tile'])
        .rename({
            "grid_lont" : "lon",
            "grid_latt" : "lat",
            "grid_lon" : "lonb",
            "grid_lat" : "latb"
        })
    )


def mappable_diag_var(
    diag_ds: xr.Dataset,
    var_name: str
):
    
    """ Converts a diagnostics dataset into a format for plotting across cubed-sphere tiles
    
    Args:
    
        diag_ds (xr.Dataset): 
            Dataset containing the variable to be plotted, along with grid spec information. Assumed to be
            `fv3_restarts.open_standard_diags` output. 
        var_name (str): 
            Name of variable to be plotted.
    
    Returns:
    
        ds (xr.Dataset):
            Dataset containing variable to be plotted as well as grid coordinates variables.
            Grid variables are renamed and ordered for plotting as first argument to `plot_cube`.
    
    Example:
    
        # plots diagnostic V850 at two times, faceted across rows
        diag_ds = fv3_restarts.open_standard_diags("/home/brianh/dev/fv3net/data/restart/C48/no_adjustment_2019-11-26")
        coastline_kwargs = {"color" : [1, 0, 0], "linewidth" : 1.5}
        axes, hs, cbar = plot_cube(
            mappable_diag_var(diag_ds, 'VGRD850').isel(time = slice(2, 4)),
            row = "time",
            coastlines = True,
            coastlines_kwargs = coastline_kwargs,
            colorbar = True,
        )
    """
    
    for var, dims in DIAG_COORD_VARS.items():
        diag_ds[var] = diag_ds[var].transpose(*dims)
        
    if 'pfull' in diag_ds[var_name].dims:
        vardims = ["grid_yt", "grid_xt", "tile", "time", "pfull"]
    elif len(diag_ds[var_name].dims) == 4:
        vardims = ["grid_yt", "grid_xt", "tile", "time"]
    else: 
        raise ValueError("Invalid variable for plotting as map.")
            
    return (
        diag_ds[[var_name]]
        .transpose(*vardims)
        .assign_coords(
            coords = {
                "lon" : diag_ds["lon"],
                "lat" : diag_ds["lat"],
                "lonb" : diag_ds['lonb'],
                "latb" : diag_ds['latb']
            }
        )
        .drop(labels = ['grid_xt', 'grid_yt', 'grid_x', 'grid_y'])

    )  


def _infer_color_limits(
    xmin: float,
    xmax: float,
    vmin: float = None,
    vmax: float = None,
    cmap: str = None
):
    
    """ "auto-magical" handling of color limits and colormap if not supplied by user
    
    Args:
    
        xmin (float): 
            Smallest value in data to be plotted
        xmax (float): 
            Largest value in data to be plotted
        vmin (float): 
            Colormap minimum value
        vmax (float): 
            Colormap minimum value
        cmap (str):
            Name of colormap
    
    Returns:
    
        vmin (float)
            Inferred colormap minimum value if not supplied, or user value if supplied.
        vmax (float)
            Inferred colormap maximum value if not supplied, or user value if supplied.
        cmap (str)
            Inferred colormap if not supplied, or user value if supplied.
    
    Example:
    
        # choose limits and cmap for data spanning 0
        >>>> _infer_color_limits(-10, 20)
        (-20, 20, 'RdBu_r')
        
    """
    

    
    if not vmin and not vmax:
        if xmin < 0 and xmax > 0:
            cmap = 'RdBu_r' if not cmap else cmap
            vabs_max = np.max([np.abs(xmin), np.abs(xmax)])
            vmin, vmax = (-vabs_max, vabs_max)
        else:
            vmin, vmax = xmin, xmax
            cmap = 'viridis' if not cmap else cmap
    elif not vmin:
        if xmin < 0 and vmax > 0:
            vmin = -vmax
            cmap = 'RdBu_r' if not cmap else cmap
        else:
            vmin = xmin
            cmap = 'viridis' if not cmap else cmap
    elif not vmax:
        if xmax > 0 and vmin < 0:
            vmax = -vmin
            cmap = 'RdBu_r' if not cmap else cmap
        else:
            vmax = xmax
            cmap = 'viridis' if not cmap else cmap
    elif not cmap:
        cmap = 'RdBu_r' if vmin == -vmax else 'viridis'
    
    return vmin, vmax, cmap


def _plot_cube_axes(
    plotting_function: str,
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    array: np.ndarray,
    ax: plt.axes,
    coastlines: bool = False,
    coastlines_kwargs: dict = None,
    title: str = None,
    **kwargs
):

    """ Plots tiled cubed sphere pcolormesh and contours for a given subplot axis,
        using np.ndarrays for all data
    
    Args:
    
        plotting_function (str):
            Function name to use in plotting the variable. Available options are 'pcolormesh' and 'contour'.
        lat (np.ndarray): 
            Array of latitudes of cell centers, of dimensions (npy, npx, tile) 
        lon (np.ndarray): 
            Array of longitudes of cell centers, of dimensions (npy, npx, tile)
        latb (np.ndarray): 
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        lonb (np.ndarray): 
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        array (np.ndarray): 
            Array of variables values at cell centers, of dimensions (npy, npx, tile)
        ax (plt.axes, optional):
            Axes onto which the map should be plotted; must be created with a cartopy projection argument. 
        coastlines (bool, optinal):
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs (dict, optional):
            Dict of options to be passed to cartopy axes's `coastline` function if `coastlines` flag is set to True.
    
    Returns:
    
        h (obj):
            matplotlib object handle associated with map subplot
    
    """
    

    if plotting_function == 'pcolormesh':
        setattr(ax, 'plotting_function', _pcolormesh_cube_arrays)
    elif plotting_function == 'contour':
        setattr(ax, 'plotting_function', _contour_cube_arrays)
    else:
        raise ValueError("Plotting functions only include pcolormesh and contour.")

    if "coastlines_kwargs" in kwargs:
        coastlines_kwargs = kwargs["coastlines_kwargs"]
        del kwargs["coastlines_kwargs"]
        
    h = ax.plotting_function(
        lat,
        lon,
        latb,
        lonb,
        array,
        ax,
        ax.projection.proj4_params['lon_0'],
        **kwargs
    )
        
    if coastlines:
        coastlines_kwargs = dict() if not coastlines_kwargs else coastlines_kwargs
        ax.coastlines(**coastlines_kwargs)
        
    if title:
        ax.set_title(title)

    return h


def _pcolormesh_cube_arrays(
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    array: np.ndarray,
    ax: plt.axes,
    central_longitude: float = 0.0,
    **kwargs
):
    
    """ Plots tiled cubed sphere pcolormesh for a given subplot axis,
        using np.ndarrays for all data
    
    Args:
        
        lat (np.ndarray): 
            Array of latitudes of cell centers, of dimensions (npy, npx, tile) 
        lon (np.ndarray): 
            Array of longitudes of cell centers, of dimensions (npy, npx, tile) 
        latb (np.ndarray): 
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        lonb (np.ndarray): 
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        array (np.ndarray): 
            Array of variables values at cell centers, of dimensions (npy, npx, tile)
        ax (plt.axes, optional):
            Axes onto which the pcolormesh should be plotted; must be created with a cartopy projection argument. 
        central_longitude (float, optional): 
            Longitude on which the projected map should be centered. Defaults to 0.0.
        **kwargs:
            Additional arguments to be passed tp `ax.pcolormesh`

    Returns:
    
        im (obj):
            `ax.pcolormesh` object handle associated with map subplot
            
    """
    
    
    
    if (len(lonb.shape) != 3) or (len(latb.shape) != 3) or (len(array.shape) != 3):
        raise ValueError('Lonb, latb, and data_var each must be 3-dimensional.')
    
    if (lonb.shape[-1] != 6) or (latb.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError('Last axis of each array must have six elements for cubed-sphere tiles.')

    if (
        (lonb.shape[0] != latb.shape[0]) or 
        (latb.shape[0] != (array.shape[0] + 1)) or 
        (lonb.shape[1] != latb.shape[1]) or 
        (latb.shape[1] != (array.shape[1] + 1))
    ):
        raise ValueError('First and second axes lengths of latb and lonb must be one greater than those of array.')
        
    if (len(lon.shape) != 3) or (len(lat.shape) != 3) or (len(array.shape) != 3):
        raise ValueError('Lonb, latb, and data_var each must be 3-dimensional.')
    
    if (lon.shape[-1] != 6) or (lat.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError('Last axis of each array must have six elements for cubed-sphere tiles.')

    if (
        (lon.shape[0] != lat.shape[0]) or 
        (lat.shape[0] != array.shape[0]) or 
        (lon.shape[1] != lat.shape[1]) or 
        (lat.shape[1] != array.shape[1])
    ):
        raise ValueError('First and second axes lengths of lat and lonb must be equal to those of array.')
    
    masked_array = np.where(
        mask_antimeridian_quads(lonb, central_longitude),
        array,
        np.nan
    )
    
    for tile in range(6):
        im = ax.pcolormesh(
            lonb[:, :, tile],
            latb[:, :, tile],
            masked_array[:, :, tile],
            transform = ccrs.PlateCarree(),
            **kwargs
        )
        
    return im


def _contour_cube_arrays(
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    array: np.ndarray,
    ax: plt.axes,
    central_longitude: float = 0.0,
    **kwargs
):
    
    """ Plots tiled cubed sphere contours for a given subplot axis,
        using np.ndarrays for all data
    
    Args:
    
        lat (np.ndarray): 
            Array of latitudes of cell centers, of dimensions (npy, npx, tile) 
        lon (np.ndarray): 
            Array of longitudes of cell centers, of dimensions (npy, npx, tile) 
        latb (np.ndarray): 
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        lonb (np.ndarray): 
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1, tile) 
        array (np.ndarray): 
            Array of variables values at cell centers, of dimensions (npy, npx, tile)
        ax (plt.axes, optional):
            Axes onto which the contours should be plotted; must be created with a cartopy projection argument. 
        central_longitude (float, optional): 
            Longitude on which the projected map should be centered. Defaults to 0.0.
        **kwargs:
            Additional arguments to be passed tp `ax.contour`

    Returns:
    
        cf (obj):
            `ax.contour` object handle associated with map subplot
            
    """
    
    if (len(lon.shape) != 3) or (len(lat.shape) != 3) or (len(array.shape) != 3):
        raise ValueError('Lonb, latb, and data_var each must be 3-dimensional.')
    
    if (lon.shape[-1] != 6) or (lat.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError('Last axis of each array must have six elements for cubed-sphere tiles.')

    if (
        (lon.shape[0] != lat.shape[0]) or 
        (lat.shape[0] != array.shape[0]) or 
        (lon.shape[1] != lat.shape[1]) or 
        (lat.shape[1] != array.shape[1])
    ):
        raise ValueError('First and second axes lengths of lat and lonb must be equal to those of array.')
        
    if (len(lonb.shape) != 3) or (len(latb.shape) != 3) or (len(array.shape) != 3):
        raise ValueError('Lonb, latb, and data_var each must be 3-dimensional.')
    
    if (lonb.shape[-1] != 6) or (latb.shape[-1] != 6) or (array.shape[-1] != 6):
        raise ValueError('Last axis of each array must have six elements for cubed-sphere tiles.')

    if (
        (lonb.shape[0] != latb.shape[0]) or 
        (latb.shape[0] != (array.shape[0] + 1)) or 
        (lonb.shape[1] != latb.shape[1]) or 
        (latb.shape[1] != (array.shape[1] + 1))
    ):
        raise ValueError('First and second axes lengths of latb and lonb must be one greater than those of array.')

    masked_array = np.where(
        mask_antimeridian_quads(lonb, central_longitude),
        array,
        np.nan
    )
    
    for tile in range(6):
        lon_tile = lon[:, :, tile]
        lon_c = np.where(lon_tile < (central_longitude + 180.0)%360.0, lon_tile, lon_tile - 360.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf = ax.contour(
                lon_c,
                lat[:, :, tile],
                masked_array[:, :, tile],
                transform = ccrs.PlateCarree(),
                **kwargs
            )
        
    return cf