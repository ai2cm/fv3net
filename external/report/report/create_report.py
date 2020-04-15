import datetime

from jinja2 import Template
from pytz import timezone

PACIFIC_TZ = "US/Pacific"
NOW_FORMAT = "%Y-%m-%d %H:%M:%S %Z"

report_html = Template(
    """
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
                <img src="{{image}}" />
            {% endfor %}
    {% endfor %}
"""
)


def create_html(report_sections, title, metadata=None):
    """Return html report of figures described in report_sections

    Args:
        report_sections (dict): description of figures to include in report. Dict with
            section names as keys and lists of figure filenames as values:
            {section name: [figure filenames in section]}
        title (str): title at top of report
        metadata (dict, optional): metadata to be printed in a table before figures.
            Defaults to None, in which case no metadata printed.

    Returns:
        (str) html report
    """
    now = datetime.datetime.now().astimezone(timezone(PACIFIC_TZ))
    html = report_html.render(
        title=title,
        sections=report_sections,
        metadata=metadata,
        now=now.strftime(NOW_FORMAT),
    )
    return html
