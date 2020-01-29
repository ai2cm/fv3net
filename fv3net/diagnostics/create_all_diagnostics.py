"""
This can be used to generate a html report of diagnostic figures using
a plot configuration yaml file.
The report will be written to output_dir/diagnostics.html where the output_dir is
one of the command line args.
"""

import argparse
import os

from jinja2 import Template
from fv3net.diagnostics.visualize import create_plot
from fv3net.diagnostics.utils import load_configs, open_dataset


report_html = Template(
    """
{% for header, image in sections.items() %}
<h2>{{header}}</h2>
<img src="{{image}}" />
{% endfor %}

"""
)


def create_diagnostics(plot_configs, data, output_dir):
    """ Create one plot per config entry in config file

    Args:
        plot_configs: List of PlotConfig objs
        data: xarray dataset
        output_dir: directory to write output figures to

    Returns:
        dict: key: header name for each figure, value: filename of figure
    """
    output_figures = {}
    for plot_config in plot_configs:
        fig = create_plot(data, plot_config)
        filename = plot_config.plot_name + ".png"
        fig.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
        output_figures[plot_config.plot_name] = filename
    return output_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="default_plot_config.json",
        help="Path for config file that describes what/how to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Location to save diagnostic plots and html summary",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to data. Can provide either a rundir (GCS or local) or a zarr.",
    )
    parser.add_argument(
        "--time-dim",
        default="initialization_time",
        help="Name of time dimension. Defaults to 'initialization_time'.",
    )
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    data = open_dataset(args.data_path)
    plot_configs = load_configs(args.config_file)
    output_figure_headings = create_diagnostics(
        plot_configs, data, args.output_dir)
    with open(f"{args.output_dir}/diagnostics.html", "w") as f:
        html = report_html.render(sections=output_figure_headings)
        f.write(html)
