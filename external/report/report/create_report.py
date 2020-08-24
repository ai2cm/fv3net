import datetime
import os
from typing import Mapping, Sequence, Union

from jinja2 import Template
from pytz import timezone

PACIFIC_TZ = "US/Pacific"
NOW_FORMAT = "%Y-%m-%d %H:%M:%S %Z"

Metadata = Mapping[str, Union[str, float, int, bool]]
Metrics = Mapping[str, Mapping[str, str]]

HTML_TEMPLATE = Template(
    """
    <html>
    <head>
        <title>{{title}}</title>
        {{header}}
    </head>

    <body>
    <h1>{{title}}</h1>
    Report created {{now}}
    {% if metadata is not none %}
        <h2>Metadata</h2>
        <table>
        {% for key, value in metadata.items() %}
            <tr>
                <th> {{ key }} </th>
                <td> {{ value }} </td>
            </tr>
        {% endfor %}
        </table>
    {% endif %}

    {% if metrics is not none %}
        <h2> Metrics </h2>
        <table border="1">
        <thead>
            <tr>
                <th> Variable </th>
                {% for key in metrics_columns %}
                    <th style="padding:10px"> {{ key }}</th>
                {% endfor %}
            </tr>
        </thead>
        {% for var, var_metrics in metrics.items() %}
        <tr>
            <th style="padding:5px">{{ var }}</th>
            {% for value in var_metrics.values() %}
                <th style="padding:3px">{{ value }}</th>
            {% endfor %}
        </tr>
        {% endfor %}
        </table>
    {% endif %}

    {% for header, images in sections.items() %}
        <h2>{{header}}</h2>
            {% for image in images %}
                {{image}}
            {% endfor %}
    {% endfor %}

    </body>
    </html>
"""
)


class ImagePlot:
    def __init__(self, path: str):
        self.path = path

    def __repr__(self) -> str:
        return f'<img src="{self.path}" />'


def resolve_plot(obj):
    if isinstance(obj, str):
        return ImagePlot(obj)
    else:
        return obj


def _save_figure(fig, filepath_relative_to_report: str, output_dir: str = None):
    """Saves figures into the directory structure expected by the report.

    Args:
        fig: matplotlib figure to save
        filepath_relative_to_report: path to save figure to, relative to
            where the report is saved (top level of output_dir if provided).
        output_dir: Directory that contains the report at the top level.
            If default None, everything is saved in working dir.
    """
    output_dir = output_dir or ""
    section_dir = os.path.dirname(filepath_relative_to_report.strip("/"))
    if not os.path.exists(os.path.join(output_dir, section_dir)):
        os.makedirs(os.path.join(output_dir, section_dir))
    fig.savefig(os.path.join(output_dir or "", filepath_relative_to_report))


def insert_report_figure(
    sections: Mapping[str, Sequence[str]],
    fig,  # matplotlib figure- omitted type hint so mpl wasn't a dependency
    filename: str,
    section_name: str,
    output_dir: str = None,
):
    """Saves figure into directory section_name in top level of output_dir
    and enters it into the report.

    Args:
        sections: Dict with section name keys and list of filenames
            (relative to the report root dir) of figures in section
        section_name: Name of report section
        output_dir: Directory to write section directories and their figures into.
            If left as default None, will write in current working directory.
            
    """
    section_dir = section_name.replace(" ", "_")
    filepath_relative_to_report = os.path.join(section_dir, filename)
    _save_figure(fig, filepath_relative_to_report, output_dir)
    sections.setdefault(section_name, []).append(filepath_relative_to_report)


def create_html(
    sections: Mapping[str, Sequence[str]],
    title: str,
    metadata: Metadata = None,
    html_header: str = None,
    metrics: Metrics = None,
) -> str:
    """Return html report of figures described in sections.

    Args:
        sections: description of figures to include in report. Dict with
            section names as keys and lists of figure filenames as values, e.g.:
            {'First section': ['figure1.png', 'figure2.png']}
        title: title at top of report
        metadata (optional): metadata to be printed in a table before figures.
            Defaults to None, in which case no metadata printed.
        html_header: string to include within the <head></head> tags of the compiled
            html file.

    Returns:
        html report
    """
    now = datetime.datetime.now().astimezone(timezone(PACIFIC_TZ))
    now_str = now.strftime(NOW_FORMAT)

    resolved_sections = {
        header: [resolve_plot(path) for path in section]
        for header, section in sections.items()
    }
    # format of metrics dict is {var: {column name: val}}
    metrics_columns = list(metrics.values())[0].keys() if metrics else None
    html = HTML_TEMPLATE.render(
        title=title,
        sections=resolved_sections,
        metadata=metadata,
        metrics=metrics,
        now=now_str,
        header=html_header,
        metrics_columns=metrics_columns,
    )
    return html
