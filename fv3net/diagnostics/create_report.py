from jinja2 import Template
import datetime


report_html = Template(
    """
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
        report_name: prepended to .html for final report file
        output_dir: dir in which figure files are located and where report html
        will be saved
        metadata (dict, optional): metadata to be printed at top of report.
            Defaults to None, in which case no metadata printed.

    Returns:
        None
    """
    with open(f"{output_dir}/{report_name}.html", "w") as f:
        html = report_html.render(
            sections=report_sections, metadata=metadata, now=datetime.datetime.now()
        )
        f.write(html)
