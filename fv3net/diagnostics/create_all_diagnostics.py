import os
from collections import defaultdict

import matplotlib.pyplot as plt
from jinja2 import Template
from vcm.diagnostic.plot import create_plot
from vcm.diagnostic.utils import load_config, read_zarr_from_gcs

IMAGES = defaultdict(list)

report_html = Template(
    """
{% for section, images in sections.items() %}
<h1>{{section}}</h1>
{% for image in images %}
<img src="{{image}}" />
{% endfor %}
{% endfor %}

"""
)


def relative_paths(paths, output_dir):
    return [os.path.relpath(path, output_dir) for path in paths]


def get_images_relative(output_dir):
    return {
        section: relative_paths(images, output_dir)
        for section, images in IMAGES.items()
    }


def create_diagnostics(plot_configs, data, output_dir):
    for plot_config in plot_configs:
        figure = create_plot(data, plot_config)
        plt.savefig(figure, os.path.join(output_dir, plot_config.plot_name + ".png"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config-file",
        description="Path for config file that describes what/how to plot",
        type=str,
        default="default_plot_config.json",
    )
    parser.add_argument(
        "output-dir",
        description="Location to save diagnostic plots and html summary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "gcs-run-dir", description="Path to remote gcs rundir", required=True
    )
    args = parser.parse_args()

    data = read_zarr_from_gcs(args.gcs_run_dir)
    plot_configs = load_config(args.config_file)
    create_diagnostics(plot_configs, data, args.output_dir)

    os.mkdir(args.output_dir)

    sections = get_images_relative(args.output_dir)
    with open(f"{args.output_dir}/diagnostics.html", "w") as f:
        html = report_html.render(sections=sections)
        f.write(html)
