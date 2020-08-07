import os
from report import __version__, create_html
from report.create_report import _save_figure, insert_report_figure


def test_version():
    assert __version__ == "0.1.0"


def test_create_html():
    title = "Report Name"
    sections = {"header": ["image.png"]}
    html = create_html(title=title, sections=sections)
    assert '<img src="image.png"' in html


class MockFig:
    # use this so that the package doesnt require matplotlib
    # dependency just for testing
    def __init__(
            self,
            content: str = "This is a test."):
        self.content = content

    def savefig(self, path):
        with open(path, "w") as f:
            print(self.content, file=f)
        

def test_insert_figure(tmpdir):
    fig = MockFig()
    report_sections = {}
    section_name = "Awesome Results Section"
    filename = "test.txt"
    
    insert_report_figure(
        report_sections, fig, filename, section_name, output_dir=tmpdir
    )
    filepath_relative_to_report = os.path.join(section_name.replace(" ", "_"), filename)
    assert report_sections == {section_name: [filepath_relative_to_report]}


def test__save_figure(tmpdir):
    fig = MockFig()
    output_dir = tmpdir
    filepath_relative_to_report = "section_dir/test.txt"
    _save_figure(fig, filepath_relative_to_report, output_dir)
    with open(os.path.join(output_dir, filepath_relative_to_report), "r") as f:
        saved_data = f.read()
    assert saved_data.replace('\n', '') == fig.content
    