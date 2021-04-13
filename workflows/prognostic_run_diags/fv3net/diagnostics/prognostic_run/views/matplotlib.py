import base64
import dataclasses
import io
import logging
from collections import defaultdict
from typing import Mapping, Sequence

import cartopy.crs as ccrs
import jinja2
import matplotlib.pyplot as plt
from fv3net.diagnostics.prognostic_run.computed_diagnostics import RunDiagnostics
import fv3viz
import xarray as xr
import pandas as pd

COORD_NAMES = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
}

_COORD_VARS = {
    "lonb": ["y_interface", "x_interface", "tile"],
    "latb": ["y_interface", "x_interface", "tile"],
    "lon": ["y", "x", "tile"],
    "lat": ["y", "x", "tile"],
}


def fig_to_b64(fig, format="png"):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format=format)
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return f"data:image/png;base64, " + pic_hash.decode()


@dataclasses.dataclass
class raw_html:
    contents: str

    def __repr__(self):
        return self.contents


template = jinja2.Template(
    """
<h2> {{varfilter}} </h2>
<table>

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


def plot_2d_matplotlib(
    run_diags: RunDiagnostics, varfilter: str, dims: Sequence, **opts
) -> str:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 2D and have the same dimensions."""

    data = defaultdict(dict)

    # kwargs handling
    ylabel = opts.pop("ylabel", "")
    x, y = dims

    variables_to_plot = [
        varname for varname in run_diags.variables if varfilter in varname
    ]

    for run in run_diags.runs:
        for varname in variables_to_plot:
            logging.info(f"plotting {varname} in {run}")
            v = run_diags.get_variable(run, varname)
            long_name_and_units = f"{v.long_name} [{v.units}]"
            fig, ax = plt.subplots()
            v.plot(ax=ax, x=x, y=y, **opts)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_title(long_name_and_units)
            plt.tight_layout()
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return raw_html(
        template.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=sorted(variables_to_plot),
            varfilter=varfilter,
        )
    )


def plot_cube_matplotlib(
    run_diags: RunDiagnostics,
    run_metrics: pd.DataFrame,
    varfilter: str,
    metric_names: Mapping[str, str] = None,
    **opts,
) -> str:
    """Plot horizontal maps of cubed-sphere data for all diagnostics whose name includes
    varfilter. All matching diagnostics must have tile, x and y dimensions
    and each dataset in run_diags must include lat/lon/latb/lonb coordinates."""

    data = defaultdict(dict)

    variables_to_plot = [
        varname for varname in run_diags.variables if varfilter in varname
    ]

    for run in run_diags.runs:
        grid_ds = xr.merge(
            [run_diags.get_variable(run, varname) for varname in _COORD_VARS]
        )
        for varname in variables_to_plot:
            logging.info(f"plotting {varname} in {run}")
            v = run_diags.get_variable(run, varname)
            ds = xr.merge([grid_ds, v])
            plot_title = _get_map_title(
                run_metrics[run_metrics.run == run], varname, metric_names
            )
            fig, ax = plt.subplots(
                figsize=(6, 3), subplot_kw={"projection": ccrs.Robinson()}
            )
            mv = fv3viz.mappable_var(ds, varname, coord_vars=_COORD_VARS, **COORD_NAMES)
            fv3viz.plot_cube(mv, ax=ax, **opts)
            ax.set_title(plot_title)
            plt.subplots_adjust(left=0.01, right=0.75, bottom=0.02)
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return raw_html(
        template.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=sorted(variables_to_plot),
            varfilter=varfilter,
        )
    )


def _get_map_title(
    metrics: pd.DataFrame, varname: str, metric_names: Mapping[str, str],
) -> str:
    metric_names = {} if metric_names is None else metric_names
    shortname = varname.split("_")[0]
    title_parts = []
    for title_name, metric_filter in metric_names.items():
        metric_name = f"{metric_filter}/{shortname}"
        metric = metrics[metrics.metric == metric_name]
        if metric.empty:
            metric_str = "[missing]"
        else:
            metric_str = f"{metric.value.item():.3f}{metric.units.item()}"
        title_parts.append(f"{title_name}: {metric_str}")
    return ", ".join(title_parts)
