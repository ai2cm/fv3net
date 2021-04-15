import base64
import dataclasses
import io
import logging
from collections import defaultdict
import numpy as np
from typing import Sequence
import xarray as xr
import textwrap

import jinja2
import matplotlib.pyplot as plt
from fv3net.diagnostics.prognostic_run.computed_diagnostics import RunDiagnostics


def fig_to_b64(fig, format="png"):
    pic_IObytes = io.BytesIO()
    fig.savefig(pic_IObytes, format=format, bbox_inches="tight")
    pic_IObytes.seek(0)
    pic_hash = base64.b64encode(pic_IObytes.read())
    return f"data:image/png;base64, " + pic_hash.decode()


@dataclasses.dataclass
class raw_html:
    contents: str

    def __repr__(self):
        return self.contents


template_image_table = jinja2.Template(
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

template_single_image_per_row = jinja2.Template(
    """
<h2> {{varfilter}} </h2>
{% for varname in variables_to_plot %}
<img src="{{ image_data[varname] }}" >
{% endfor %}
</table>
"""
)


CBAR_RANGE = {
    "eastward_wind_pressure_level_zonal_bias": 30,
    "northward_wind_pressure_level_zonal_bias": 3,
    "air_temperature_pressure_level_zonal_bias": 15,
    "specific_humidity_pressure_level_zonal_bias": 1e-3,
    "vertical_wind_pressure_level_zonal_bias": 0.02,
}


def _data_array_from_run_diags(run_diags, var):
    values, run_coords = [], []
    for run in run_diags.runs:
        da = run_diags.get_variable(run, var)
        if not np.isnan(da.mean(skipna=True)):
            values.append(da)
            run_coords.append(run)
    return xr.concat(values, dim="run").assign_coords({"run": run_coords})


def plot_2d_matplotlib_groupby_run(
    run_diags: RunDiagnostics,
    varfilter: str,
    contours: bool = False,
    dims: Sequence = None,
    **opts,
):
    """Plot all diagnostics whose name includes varfilter. Runs containing
    the variable  of interest are combined into a facet plot with a common
    color scale. All matching diagnostics must be 2D and have the
    same dimensions."""
    data = defaultdict(dict)

    # kwargs handling
    levels = opts.pop("levels", 75)

    # ignore the symmetric option
    opts.pop("symmetric", None)
    x, y = dims
    variables_to_plot = [
        varname for varname in run_diags.variables if varfilter in varname
    ]

    for varname in variables_to_plot:
        logging.info(f"plotting {varname}")
        da = _data_array_from_run_diags(run_diags, varname)
        figsize = (len(da.run) * 4.5, 4)

        long_name_and_units = f"{da.long_name} [{da.units}]"

        vmax = CBAR_RANGE.get(varname)
        robust = True if vmax else False
        if contours:
            faceted = da.plot.contour(
                x=x,
                y=y,
                col="run",
                levels=levels,
                add_colorbar=True,
                figsize=figsize,
                vmax=vmax,
                robust=robust,
                **opts,
            )
        else:
            faceted = da.plot(
                x=x,
                y=y,
                col="run",
                add_colorbar=True,
                figsize=figsize,
                vmax=vmax,
                robust=robust,
                **opts,
            )
        for i, ax in enumerate(faceted.axes.flat):
            ax.set_title("\n".join(textwrap.wrap(da.run.values[i], 35)))
        plt.suptitle(long_name_and_units, y=1.04)
        data[varname] = fig_to_b64(faceted.fig)
        plt.close(faceted.fig)
    return raw_html(
        template_single_image_per_row.render(
            image_data=data, variables_to_plot=variables_to_plot, varfilter=varfilter,
        )
    )


def plot_2d_matplotlib_individual_runs(
    run_diags: RunDiagnostics, varfilter: str, dims: Sequence = None, **opts
) -> str:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 2D and have the same dimensions."""

    data = defaultdict(dict)

    # kwargs handling
    ylabel = opts.pop("ylabel", "")
    invert_yaxis = opts.pop("invert_yaxis", False)
    # ignore the symmetric option
    opts.pop("symmetric", None)
    x, y = dims

    variables_to_plot = [
        varname for varname in run_diags.variables if varfilter in varname
    ]

    for run in run_diags.runs:
        for varname in variables_to_plot:
            logging.info(f"plotting {varname} in {run}")
            v = run_diags.get_variable(run, varname).rename("value")
            if dims is None:
                dims = list(v.dims)
            long_name_and_units = f"{v.long_name} [{v.units}]"
            fig, ax = plt.subplots()
            v.plot(ax=ax, x=x, y=y, yincrease=not invert_yaxis, **opts)
            if ylabel:
                ax.set_ylabel(ylabel)
            ax.set_title(long_name_and_units)
            plt.tight_layout()
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return raw_html(
        template_image_table.render(
            image_data=data,
            runs=sorted(run_diags.runs),
            variables_to_plot=variables_to_plot,
            varfilter=varfilter,
        )
    )
