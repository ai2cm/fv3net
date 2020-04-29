import datetime
from typing import Mapping, Sequence, Union

from jinja2 import Template
from pytz import timezone

PACIFIC_TZ = "US/Pacific"
NOW_FORMAT = "%Y-%m-%d %H:%M:%S %Z"

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


class Plot:
    @staticmethod
    def create(obj) -> "Plot":
        """Factory method for creating different kinds of plots
        """
        if isinstance(obj, Plot):
            return obj

        return ImagePlot(obj)

    def render(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.render()


class ImagePlot(Plot):
    def __init__(self, path: str):
        self.path = path

    def render(self) -> str:
        return f'<img src="{self.path}" />'


class HTMLPlot(Plot):
    def __init__(self, html: str):
        self.html = html

    def render(self):
        return self.html


def create_html(
    sections: Mapping[str, Sequence[str]],
    title: str,
    metadata: Mapping[str, Union[str, float, int, bool]] = None,
    html_header: str = None,
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
        header: [Plot.create(path) for path in section]
        for header, section in sections.items()
    }

    html = HTML_TEMPLATE.render(
        title=title,
        sections=resolved_sections,
        metadata=metadata,
        now=now_str,
        header=html_header,
    )
    return html
