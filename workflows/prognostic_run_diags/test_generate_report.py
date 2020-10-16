from generate_report import (
    upload,
    _assign_plot_groups,
    detect_rundirs,
    _html_link,
    get_movie_links,
)

import pytest
import fsspec
import uuid
import os
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


@pytest.mark.parametrize(
    ["rundir_list", "expected"],
    [
        pytest.param(
            ["blah-blah-baseline"],
            [
                {
                    "run": "blah-blah-baseline",
                    "plot_group": "baseline",
                    "group_ind": -1,
                    "within_group_ind": 0,
                }
            ],
            id="baseline",
        ),
        pytest.param(
            ["blah-blah-baseline", "test-config"],
            [
                {
                    "run": "blah-blah-baseline",
                    "plot_group": "baseline",
                    "group_ind": -1,
                    "within_group_ind": 0,
                },
                {
                    "run": "test-config",
                    "plot_group": "test-config",
                    "group_ind": 0,
                    "within_group_ind": 0,
                },
            ],
            id="baseline_plus_one_single",
        ),
        pytest.param(
            ["blah-blah-baseline", "test-config-seed-0", "test-config-seed-1"],
            [
                {
                    "run": "blah-blah-baseline",
                    "plot_group": "baseline",
                    "group_ind": -1,
                    "within_group_ind": 0,
                },
                {
                    "run": "test-config-seed-0",
                    "plot_group": "test-config",
                    "group_ind": 0,
                    "within_group_ind": 0,
                },
                {
                    "run": "test-config-seed-1",
                    "plot_group": "test-config",
                    "group_ind": 0,
                    "within_group_ind": 1,
                },
            ],
            id="baseline_plus_one_group",
        ),
        pytest.param(
            [
                "blah-blah-baseline",
                "other-config",
                "test-config-seed-0",
                "test-config-seed-1",
            ],
            [
                {
                    "run": "blah-blah-baseline",
                    "plot_group": "baseline",
                    "group_ind": -1,
                    "within_group_ind": 0,
                },
                {
                    "run": "other-config",
                    "plot_group": "other-config",
                    "group_ind": 0,
                    "within_group_ind": 0,
                },
                {
                    "run": "test-config-seed-0",
                    "plot_group": "test-config",
                    "group_ind": 1,
                    "within_group_ind": 0,
                },
                {
                    "run": "test-config-seed-1",
                    "plot_group": "test-config",
                    "group_ind": 1,
                    "within_group_ind": 1,
                },
            ],
            id="baseline_plus_one_group_plus_another",
        ),
    ],
)
def test__assign_plot_groups(rundir_list, expected):
    out = _assign_plot_groups(rundir_list)
    assert out == expected


def test_detect_rundirs(tmpdir):

    fs = fsspec.filesystem("file")

    rundirs = ["rundir1", "rundir2"]
    for rdir in rundirs:
        tmpdir.mkdir(rdir).join("diags.nc").write("foobar")

    tmpdir.mkdir("not_a_rundir").join("useless_file.txt").write("useless!")

    result = detect_rundirs(tmpdir, fs)

    assert len(result) == 2
    for found_dir in result:
        assert found_dir in rundirs


def test_detect_rundirs_fail_less_than_2(tmpdir):

    fs = fsspec.filesystem("file")

    tmpdir.mkdir("rundir1").join("diags.nc").write("foobar")

    with pytest.raises(ValueError):
        detect_rundirs(tmpdir, fs)


def test__html_links():
    tag = "my-tag"
    url = "http://www.domain.com"
    expected = "<a href='http://www.domain.com'>my-tag</a>"
    assert expected == _html_link(url, tag)


def test_get_movie_links(tmpdir):
    fs = fsspec.filesystem("file")
    domain = "http://www.domain.com"
    rdirs = ["rundir1", "rundir2"]
    for rdir in rdirs:
        tmpdir.mkdir(rdir).join("movie1.mp4").write("foobar")
    tmpdir.join(rdirs[0]).join("movie2.mp4").write("foobar")

    result = get_movie_links(tmpdir, rdirs, fs, domain=domain)

    assert "movie1.mp4" in result
    assert "movie2.mp4" in result
    assert os.path.join(domain, tmpdir, rdirs[0], "movie2.mp4") in result["movie2.mp4"]
