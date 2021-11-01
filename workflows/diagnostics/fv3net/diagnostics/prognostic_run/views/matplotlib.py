import base64
import io
import logging
from collections import defaultdict
from typing import Tuple, Mapping
import xarray as xr

import cartopy.crs as ccrs
import jinja2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    RunDiagnostics,
    RunMetrics,
)
import fv3viz
from report import RawHTML

COORD_VARS = ["lon", "lat", "lonb", "latb"]


def fig_to_b64(fig, format="png", dpi=None):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format=format, bbox_inches="tight", dpi=dpi)
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return f"data:image/png;base64, " + pic_hash.decode()


template = jinja2.Template(
    """
<h2> {{varfilter}} </h2>
<table cellpadding="0" cellspacing="0">

<tr>
{% for run in runs %}
<th><center> {{ run }} </center></th>
{% endfor %}
</tr>

{% for varname in variables_to_plot %}
<tr>
{% for run in runs %}
<td>
<img src="{{ image_data[varname][run] }}" width="500px" />
</td>
{% endfor %}
</tr>
{% endfor %}
</table>
"""
)

CONTOUR_LEVELS = {
    "eastward_wind_pressure_level_zonal_bias": np.arange(-30, 31, 4),
    "northward_wind_pressure_level_zonal_bias": np.arange(-3, 3.1, 0.4),
    "air_temperature_pressure_level_zonal_bias": np.arange(-15, 16, 2),
    "specific_humidity_pressure_level_zonal_bias": np.arange(-1.1e-3, 1.2e-3, 2e-4),
    "vertical_wind_pressure_level_zonal_bias": np.arange(-2.1e-2, 2.2e-2, 2e-3),
    "mass_streamfunction_pressure_level_zonal_bias": np.arange(-105, 106, 10),
}


def plot_2d_matplotlib(
    run_diags: RunDiagnostics,
    varfilter: str,
    dims: Tuple[str, str],
    contour=False,
    **opts,
) -> RawHTML:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 2D and have the same dimensions."""

    data = defaultdict(dict)

    # kwargs handling
    ylabel = opts.pop("ylabel", "")
    x, y = dims

    variables_to_plot = run_diags.matching_variables(varfilter)
    for varname in variables_to_plot:
        if not contour:
            opts["vmin"], opts["vmax"], opts["cmap"] = _get_cmap_kwargs(
                run_diags, varname, robust=False
            )
        for run in run_diags.runs:
            logging.info(f"plotting {varname} in {run}")
            v = run_diags.get_variable(run, varname)
            long_name_and_units = f"{v.long_name} [{v.units}]"
            fig, ax = plt.subplots()
            if contour:
                levels = CONTOUR_LEVELS.get(varname)
                xr.plot.contourf(
                    v, ax=ax, x=x, y=y, levels=levels, extend="both", **opts
                )
            else:
                v.plot(ax=ax, x=x, y=y, **opts)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_title(long_name_and_units)
            plt.tight_layout()
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return RawHTML(
        template.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=sorted(variables_to_plot),
            varfilter=varfilter,
        )
    )


def plot_cubed_sphere_map(
    run_diags: RunDiagnostics,
    run_metrics: RunMetrics,
    varfilter: str,
    metrics_for_title: Mapping[str, str] = None,
) -> str:
    """Plot horizontal maps of cubed-sphere data for diagnostics which match varfilter.
    
    Args:
        run_diags: the run diagnostics
        run_metrics: the run metrics, which can be used to annotate plots
        varfilter: pattern to filter variable names
        metrics_for_title: metrics to put in plot title. Mapping from label to use in
            plot title to metric_type.

    Note:
        All matching diagnostics must have tile, x and y dimensions and each dataset in
        run_diags must include lat/lon/latb/lonb coordinates.
    """

    data = defaultdict(dict)
    if metrics_for_title is None:
        metrics_for_title = {}

    variables_to_plot = run_diags.matching_variables(varfilter)
    for varname in variables_to_plot:
        vmin, vmax, cmap = _get_cmap_kwargs(run_diags, varname, robust=True)
        for run in run_diags.runs:
            logging.info(f"plotting {varname} in {run}")
            shortname = varname.split(varfilter)[0][:-1]
            ds = run_diags.get_variables(run, COORD_VARS + [varname])
            plot_title = _render_map_title(
                run_metrics, shortname, run, metrics_for_title
            )
            fig, ax = plt.subplots(
                figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
            )
            fv3viz.plot_cube(ds, varname, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(plot_title)
            plt.subplots_adjust(left=0.01, right=0.75, bottom=0.02)
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return RawHTML(
        template.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=sorted(variables_to_plot),
            varfilter=varfilter,
        )
    )


def plot_histogram(
    run_diags: RunDiagnostics, varname: str, xscale="linear", yscale="linear"
) -> RawHTML:
    """Plot 1D histogram of varname overlaid across runs."""

    logging.info(f"plotting {varname}")
    fig, ax = plt.subplots()
    bin_name = varname.replace("histogram", "bins")
    for run in run_diags.runs:
        v = run_diags.get_variable(run, varname)
        ax.step(v[bin_name], v, label=run, where="post", linewidth=1)
    ax.set_xlabel(f"{v.long_name} [{v.units}]")
    ax.set_ylabel(f"Frequency [({v.units})^-1]")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim([v[bin_name].values[0], v[bin_name].values[-1]])
    ax.legend()
    fig.tight_layout()
    data = fig_to_b64(fig, dpi=150)
    plt.close(fig)
    return RawHTML(f'<img src="{data}" width="800px" />')


def plot_histogram2d(
    run_diags: RunDiagnostics, xname: str, yname: str, conditional: bool = False
) -> RawHTML:
    """Plot 2D histogram of xname versus yname overlaid across runs."""

    data = defaultdict(dict)
    count_name = f"{xname.lower()}_versus_{yname.lower()}_hist_2d"
    conditional_average_name = f"conditional_average_of_{yname}_on_{xname}"
    x_bin_name = f"{xname}_bins"
    y_bin_name = f"{yname}_bins"
    x_bin_widths_name = f"{xname.lower()}_bin_width_hist_2d"
    y_bin_widths_name = f"{yname.lower()}_bin_width_hist_2d"

    for run in run_diags.runs:
        logging.info(f"plotting {xname} versus {yname} 2D histogram for {run}.")

        count = run_diags.get_variable(run, count_name)
        if conditional:
            count = count / count.sum(y_bin_name)
            count = count / count.max(y_bin_name)
        conditional_average = run_diags.get_variable(run, conditional_average_name)
        x_bin_widths = run_diags.get_variable(run, x_bin_widths_name)
        y_bin_widths = run_diags.get_variable(run, y_bin_widths_name)
        x = x_bin_widths[x_bin_name]
        y = y_bin_widths[y_bin_name]
        xedges = np.append(x.values, x.values[-1] + x_bin_widths.values[-1])
        yedges = np.append(y.values, y.values[-1] + y_bin_widths.values[-1])
        xcenters = x.values + 0.5 * x_bin_widths.values
        fig, ax = plt.subplots()
        xx, yy = np.meshgrid(xedges, yedges)
        ax.pcolormesh(xx, yy, count.T, norm=matplotlib.colors.LogNorm())
        ax.plot(xcenters, conditional_average, color="r", linewidth=2)
        ax.set_xlabel(f"{xname} [{x_bin_widths.units}]")
        ax.set_ylabel(f"{yname} [{y_bin_widths.units}]")
        ax.set_xlim([xedges[0], xedges[-1]])
        ax.set_ylim([yedges[0], yedges[-1]])
        plt.tight_layout()
        data[count_name][run] = fig_to_b64(fig)
        plt.close(fig)
    return RawHTML(
        template.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=[count_name],
            varfilter="2D Histogram",
        )
    )


def _render_map_title(
    metrics: RunMetrics, variable: str, run: str, metrics_for_title: Mapping[str, str],
) -> str:
    title_parts = []
    for name_in_figure_title, metric_type in metrics_for_title.items():
        metric_value = metrics.get_metric_value(metric_type, variable, run)
        metric_units = metrics.get_metric_units(metric_type, variable, run)
        title_parts.append(f"{name_in_figure_title}: {metric_value:.3f} {metric_units}")
    return ", ".join(title_parts)


def _get_cmap_kwargs(run_diags, variable, **kwargs):
    input_data = []
    for run in run_diags.runs:
        input_data.append(run_diags.get_variable(run, variable).assign_coords(run=run))
    input_data = xr.concat(input_data, dim="run")
    return fv3viz.infer_cmap_params(input_data.values, **kwargs)
