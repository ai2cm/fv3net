"""
Some helper function for visualization.
"""
import holoviews as hv


def make_image(
    sliced_data,
    cmap_range=None,
    coords=None,
    quad=False,
    invert_y=False,
    **kwargs
):
    coords = coords or (
        ["grid_xt", "grid_yt"]
        if "grid_xt" in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_img = (
        hv.QuadMesh(sliced_data, coords)
        if quad
        else hv.Image(sliced_data, coords)
    )
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(
            **{var_name: hv.Dimension(var_name, range=cmap_range)}
        )
    hv_img = hv_img.options(**kwargs)
    return hv_img


def make_animation(
    sliced_data,
    cmap_range=None,
    coords=None,
    quad=False,
    invert_y=False,
    **kwargs
):
    coords = coords or (
        ["grid_xt", "grid_yt"]
        if "grid_xt" in sliced_data.coords
        else ["lon", "lat"]
    )
    hv_ds = hv.Dataset(sliced_data)
    hv_img = hv_ds.to(hv.QuadMesh if quad else hv.Image, coords).options(
        **kwargs
    )
    if cmap_range is not None:
        var_name = sliced_data.name
        hv_img = hv_img.redim(
            **{var_name: hv.Dimension(var_name, range=cmap_range)}
        )
    return hv_img


def plot_cube(
        data: xr.DataArray,
        grid: xr.Dataset,
        ax=None,
        colorbar=True,
        contours=False,
        **kwargs):
    """ Plots cubed sphere grids into a global map projection

    Arguments:

    Data: Dataarray of variable to plot assumed to have dimensions (grid_ty, grid_xt and tile)

    grid: Dataset of grid variables that must include:
        -lat, lon, latb, lonb (where lat and lon are grid centers and lonb and latb are grid edges)

    Returns:

    Fig and ax handles

    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 7))
    if 'grid_y' in data.dims or 'grid_x' in data.dims:
        data = shift_edge_var_to_center(data)

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
        if contours:
            cf = ax.contour(
                grid.grid_lont.isel(tile=tile),
                grid.grid_latt.isel(tile=tile),
                masked.isel(tile=tile),
                transform=ccrs.PlateCarree(),
                levels=[0],
                linewidths=0.5
            )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(data.name)
    ax.coastlines(color=[0, 0.25, 0], linewidth=1.5)

    return fig, ax