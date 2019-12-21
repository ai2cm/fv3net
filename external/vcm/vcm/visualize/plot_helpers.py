import numpy as np


def _infer_color_limits(
    xmin: float, xmax: float, vmin: float = None, vmax: float = None, cmap: str = None
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

    return vmin, vmax, cmap


def _get_var_label(attrs: dict, var_name: str):

    """ Get the label for the variable on the colorbar

    Args:

        attrs (dict):
            Variable aattribute dict
        var_name (str):
            Short name of variable

    Returns:

        var_name (str)
            Variable name to be plotted, either short name or annotated name


    """

    if "long_name" in attrs and "units" in attrs:
        return f"{attrs['long_name']} [{attrs['units']}]"
    else:
        return var_name
