from jinja2 import Template
import datetime
from pytz import timezone

PACIFIC_TZ = "US/Pacific"
NOW_FORMAT = "%Y-%m-%d %H:%M:%S %Z"

report_html = Template(
    """
    <h1>{{report_name}}</h1>
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


def create_report(report_sections, report_name, output_dir, metadata=None):
    """

    Args:
        report_sections: dict {section name: [figure filenames in section]}
        report_name: prepended to .html for final report file and used as a title
            at top of report with underscores replaced by spaces
        output_dir: dir in which figure files are located and where report html
        will be saved
        metadata (dict, optional): metadata to be printed at top of report.
            Defaults to None, in which case no metadata printed.

    Returns:
        None
    """
    now = datetime.datetime.now().astimezone(timezone(PACIFIC_TZ))
    with open(f"{output_dir}/{report_name}.html", "w") as f:
        html = report_html.render(
            report_name=report_name.replace("_", " "),
            sections=report_sections,
            metadata=metadata,
            now=now.strftime(NOW_FORMAT),
        )
        f.write(html)
