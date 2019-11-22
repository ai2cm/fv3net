!/usr/bin/env python
import os
import sys
from jinja2 import Template
from collections import defaultdict

from src.data import SAMRun, open_ngaqua

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
    return {section: relative_paths(images, output_dir) for section, images in
            IMAGES.items()}


def create_diagnostics(
        schema,
        data
):
    pass




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "timestep-schema",
        description="Expected schema of rundir data timesteps",
        type=str,
        choices=['single', 'multiple'],
        required=True
    )
    parser.add_argument(
        "output-dir",
        description="Location to save diagnostic plots and html summary",
        type=str,
        required=True
    )
    # keep or remove this? plotting routines
    parser.add_argument(
        "rundir",
        description="Path to rundir",
        required=False
    )
    args = parser.parse_args()
    create_diagnostics(args.timestep_schema)


    os.mkdir(args.output_dir)

    sections = get_images_relative(args.output_dir)
    with open(f"{args.output_dir}/diagnostics.html", "w") as f:
        html = report_html.render(sections=sections)
        f.write(html)