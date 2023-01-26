from typing import Optional, Tuple

import numpy as np
import textwrap
from matplotlib.colors import ListedColormap, from_levels_and_colors
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator


def _align_grid_var_dims(da, required_dims):
    missing_dims = set(required_dims).difference(da.dims)
    if len(missing_dims) > 0:
        raise ValueError(
            f"Grid variable {da.name} missing dims {missing_dims}. "
            "Incompatible grid metadata may have been passed."
        )
    redundant_dims = set(da.dims).difference(required_dims)
    if len(redundant_dims) == 0:
        da_out = da.transpose(*required_dims)
    else:
        redundant_dims_index = {dim: 0 for dim in redundant_dims}
        da_out = (
            da.isel(redundant_dims_index)
            .drop_vars(redundant_dims, errors="ignore")
            .transpose(*required_dims)
        )
    return da_out


def _align_plot_var_dims(da, coord_y_center, coord_x_center):
    first_dims = [coord_y_center, coord_x_center, "tile"]
    missing_dims = set(first_dims).difference(set(da.dims))
    if len(missing_dims) > 0:
        raise ValueError(
            f"Data array to be plotted {da.name} missing dims {missing_dims}. "
            "Incompatible grid metadata may have been passed."
        )
    rest = set(da.dims).difference(set(first_dims))
    xpose_dims = first_dims + list(rest)
    return da.transpose(*xpose_dims)


def _min_max_from_percentiles(x, min_percentile=2, max_percentile=98):
    """ Use +/- small percentile to determine bounds for colorbar. Avoids the case
    where an outlier in the data causes the color scale to be washed out.

    Args:
        x: array of data values
        min_percentile: lower percentile to use instead of absolute min
        max_percentile: upper percentile to use instead of absolute max

    Returns:
        Tuple of values at min_percentile, max_percentile
    """
    x = np.array(x).flatten()
    x = x[~np.isnan(x)]
    if len(x) == 0:
        # all values of x are equal to np.nan
        xmin, xmax = np.nan, np.nan
    else:
        xmin, xmax = np.percentile(x, [min_percentile, max_percentile])
    return xmin, xmax


def _color_palette(cmap, n_colors):
    colors_i = np.linspace(0, 1.0, n_colors)
    if isinstance(cmap, (list, tuple)):
        cmap = ListedColormap(cmap, N=n_colors)
        pal = cmap(colors_i)
    elif isinstance(cmap, str):
        pal = get_cmap(cmap)(colors_i)
    else:
        pal = cmap(colors_i)

    return pal


def _determine_extend(xmin, xmax, vmin, vmax):
    extend_min = xmin < vmin
    extend_max = xmax > vmax
    if extend_min and extend_max:
        return "both"
    elif extend_min:
        return "min"
    elif extend_max:
        return "max"
    else:
        return "neither"


def _infer_color_limits(
    xmin: float,
    xmax: float,
    vmin: float = None,
    vmax: float = None,
    cmap: str = None,
    levels: Optional[int] = None,
):
    """ "auto-magical" handling of color limits and colormap if not supplied by
    user

    Args:
        xmin (float):
            Smallest value in data to be plotted
        xmax (float):
            Largest value in data to be plotted
        vmin (float, optional):
            Colormap minimum value. Default None.
        vmax (float, optional):
            Colormap minimum value. Default None.
        cmap (str, optional):
            Name of colormap. Default None.
        levels:
            Experimental based on xarray plot levels argument.  Divide pcolormesh color
            map into descrete colors.

    Returns:
        vmin (float)
            Inferred colormap minimum value if not supplied, or user value if
            supplied.
        vmax (float)
            Inferred colormap maximum value if not supplied, or user value if
            supplied.
        cmap (str)
            Inferred colormap if not supplied, or user value if supplied.

    Example:
        # choose limits and cmap for data spanning 0
        >>>> _infer_color_limits(-10, 20)
        (-20, 20, 'RdBu_r')
    """
    if vmin is None and vmax is None:
        if xmin < 0 and xmax > 0:
            cmap = "RdBu_r" if not cmap else cmap
            vabs_max = np.max([np.abs(xmin), np.abs(xmax)])
            vmin, vmax = (-vabs_max, vabs_max)
        else:
            vmin, vmax = xmin, xmax
            cmap = "viridis" if not cmap else cmap
    elif vmin is None:
        if xmin < 0 and vmax > 0:
            vmin = -vmax
            cmap = "RdBu_r" if not cmap else cmap
        else:
            vmin = xmin
            cmap = "viridis" if not cmap else cmap
    elif vmax is None:
        if xmax > 0 and vmin < 0:
            vmax = -vmin
            cmap = "RdBu_r" if not cmap else cmap
        else:
            vmax = xmax
            cmap = "viridis" if not cmap else cmap
    elif not cmap:
        cmap = "RdBu_r" if vmin == -vmax else "viridis"

    extend = _determine_extend(xmin, xmax, vmin, vmax)

    if levels is not None:
        if extend == "both":
            ext_n = 2
        elif extend in ["min", "max"]:
            ext_n = 1
        else:
            ext_n = 0

        # N in MaxNLocator refers to bins, not ticks
        ticker = MaxNLocator(levels - 1)
        levels = ticker.tick_values(vmin, vmax)
        vmin, vmax = levels[0], levels[-1]
        n_colors = len(levels) + ext_n - 1
        pal = _color_palette(cmap, n_colors)
        cmap, cnorm = from_levels_and_colors(levels, pal, extend=extend)

    return vmin, vmax, cmap, extend


def _get_var_label(attrs: dict, var_name: str, max_line_length: int = 30):
    """ Get the label for the variable on the colorbar

    Args:
        attrs (dict):
            Variable aattribute dict
        var_name (str):
            Short name of variable
        max_line_length (int, optional):
            Max number of characters on each line of returned label.
            Defaults to 30.

    Returns:
        var_label (str)
            long_name [units], var_name [units] or var_name depending on attrs
    """
    if "long_name" in attrs:
        var_label = attrs["long_name"]
    else:
        var_label = var_name
    if "units" in attrs:
        var_label += f" [{attrs['units']}]"
    return "\n".join(textwrap.wrap(var_label, max_line_length))


def infer_cmap_params(
    data: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    robust: bool = False,
    levels: Optional[int] = None,
) -> Tuple[float, float, str]:
    """Determine useful colorbar limits and cmap for given data.

    Args:
        data: The data to be plotted.
        vmin: Optional minimum for colorbar.
        vmax: Optional maximum for colorbar.
        cmap: Optional colormap to use.
        robust: If true, use 2nd and 98th percentiles for colorbar limits.

    Returns:
        Tuple of (vmin, vmax, cmap).
    """
    if robust:
        xmin, xmax = _min_max_from_percentiles(data)
    else:
        xmin, xmax = np.nanmin(data), np.nanmax(data)
    vmin, vmax, cmap, extend = _infer_color_limits(xmin, xmax, vmin, vmax, cmap, levels)
    return vmin, vmax, cmap, extend
