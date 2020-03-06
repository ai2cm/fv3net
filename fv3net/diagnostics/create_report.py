import fsspec
from jinja2 import Template


report_html = Template(
    """
    {% for header, images in sections.items() %}
        <h2>{{header}}</h2>
            {% for image in images %}
                <img src="{{image}}" />
            {% endfor %}
    {% endfor %}
"""
)


def create_report(report_sections, report_name, output_dir):
    """

    Args:
        report_sections: dict {section name: [figure filenames in section]}
        report_name: prepended to .html for final report file
        output_dir: dir in which figure files are located and where report html
        will be saved

    Returns:
        None
    """
    fs, _, _ = fsspec.get_fs_token_paths(output_dir)
    with fs.open(f"{output_dir}/{report_name}.html", "w") as f:
        html = report_html.render(sections=report_sections)
        f.write(html)
