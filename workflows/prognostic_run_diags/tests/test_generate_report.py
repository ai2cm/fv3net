from fv3net.diagnostics.prognostic_run.views.static_report import (
    upload,
    _html_link,
)
from fv3net.diagnostics.prognostic_run.views.static_report import render_links

import pytest
import uuid
from google.cloud.storage.client import Client


@pytest.fixture()
def client():
    return Client()


def test_upload_html_local(tmpdir):
    url = str(tmpdir.join("index.html"))
    html = "hello"
    mime = "text/html"
    upload(html, url, mime)

    with open(url) as f:
        assert html == f.read()


def test_upload_html_gcs(client: Client):
    id_ = str(uuid.uuid4())
    html = "hello"
    mime = "text/html"
    bucket_name = "vcm-ml-scratch"
    blob = f"testing/{id_}/index.html"
    url = f"gs://{bucket_name}/{blob}"

    upload(html, url, mime)
    bucket = client.bucket(bucket_name)
    blob = bucket.get_blob(blob)
    blob.content_type == "text/html"


def test__html_links():
    tag = "my-tag"
    url = "http://www.domain.com"
    expected = "<a href='http://www.domain.com'>my-tag</a>"
    assert expected == _html_link(url, tag)


def test_render_links(regtest):
    link_dict = {
        "column_heating_moistening.mp4": [
            (
                "/Users/noah/workspace/VulcanClimateModeling/fv3net/workflows/prognostic_run_diags/45adebaeb631/run1/column_heating_moistening.mp4",  # noqa
                "run1",
            ),
            (
                "/Users/noah/workspace/VulcanClimateModeling/fv3net/workflows/prognostic_run_diags/45adebaeb631/run2/column_heating_moistening.mp4",  # noqa
                "run2",
            ),
        ]
    }
    output = render_links(link_dict)
    for val in output.values():
        assert isinstance(val, str)
    print(output, file=regtest)
