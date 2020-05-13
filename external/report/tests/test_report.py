from report import __version__, create_html


def test_version():
    assert __version__ == "0.1.0"


def test_create_html():
    title = "Report Name"
    sections = {"header": ["image.png"]}
    html = create_html(title=title, sections=sections)
    assert '<img src="image.png"' in html
