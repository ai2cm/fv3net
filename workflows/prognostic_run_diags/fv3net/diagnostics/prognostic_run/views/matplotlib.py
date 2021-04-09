import base64
import dataclasses
import io
import logging
from collections import defaultdict
from typing import Iterable, Sequence

import jinja2
import matplotlib.pyplot as plt
import xarray as xr


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
<td><center> {{ run }} </center></td>
{% endfor %}
</tr>

{% for varname, runs in diagnostics.items() %}
{% for run, src in runs.items() %}
<td>
<img src="{{src}}" width="500px" />
</td>
{% endfor %}
</tr>
{% endfor %}
</table>
"""
)


def plot_2d_matplotlib(
    diagnostics: Iterable[xr.Dataset], varfilter: str, dims: Sequence = None, **opts
) -> str:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 2D and have the same dimensions."""

    data = defaultdict(dict)

    ylabel = opts.pop("ylabel", "")
    invert_yaxis = opts.pop("invert_yaxis", False)
    # ignore the symmetric option
    opts.pop("symmetric", None)
    x, y = dims
    runs = set()

    for ds in diagnostics:
        run = ds.attrs["run"]
        runs.add(run)
        variables_to_plot = [varname for varname in ds if varfilter in varname]
        for varname in variables_to_plot:
            logging.info(f"plotting {varname} in {run}")
            v = ds[varname].rename("value")
            if dims is None:
                dims = list(v.dims)
            long_name_and_units = f"{v.long_name} [{v.units}]"
            fig, ax = plt.subplots()
            v.plot(ax=ax, x=x, y=y, yincrease=not invert_yaxis, **opts)
            ax.set_ylabel(ylabel)
            ax.set_title(long_name_and_units)
            data[varname][run] = fig_to_b64(fig)
            plt.close(fig)
    return raw_html(template.render(diagnostics=data, runs=runs, varfilter=varfilter))
