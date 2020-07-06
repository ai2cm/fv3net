from generate_report import upload, _parse_metadata, detect_rundirs

import pytest
from google.cloud.storage.client import Client
import uuid


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


def test__parse_metadata():
    run = "blah-blah-baseline"
    out = _parse_metadata(run)
    assert out == {"run": run, "baseline": "Baseline", "one_step": "blah-blah"}


def test_detect_rundirs(tmpdir):

    dir1 = tmpdir.mkdir("rundir1").join("diags.nc")
    dir2 = tmpdir.mkdir("rundir2").join("diags.nc")
    dir3 = tmpdir.mkdir("not_a_rundir").join("useless_file.txt")

    dir1.write("foobar")
    dir2.write("foobar")
    dir3.write("I'm useless!")

    expected = [str(dir1), str(dir2)]
    result = detect_rundirs(tmpdir)

    assert len(result) == 2
    for found_dir in result:
        assert found_dir in expected


def test_detect_rundirs_fail_less_than_2(tmpdir):

    tmpdir.mkdir("rundir1").join("diags.nc").write("foobar")

    with pytest.raises(ValueError):
        detect_rundirs(tmpdir)
