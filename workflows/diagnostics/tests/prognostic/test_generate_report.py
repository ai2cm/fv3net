import uuid

import numpy as np
import pandas as pd
import pytest
import xarray
from google.cloud.storage.client import Client

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    RunDiagnostics,
    RunMetrics,
)
from fv3net.diagnostics.prognostic_run.views.matplotlib import (
    plot_2d_matplotlib,
    _get_cmap_kwargs,
)
from fv3net.diagnostics.prognostic_run.views.static_report import (
    _html_link,
    render_links,
    upload,
    _get_metric_type_df,
    _get_metric_df,
    _get_movie_manifest,
    _get_public_links,
)


METRICS_DF = pd.DataFrame(
    {
        "run": ["run1", "run1", "run2", "run2"],
        "baseline": [False, False, True, True],
        "metric": [
            "rmse_of_time_mean/precip",
            "rmse_of_time_mean/h500",
            "rmse_of_time_mean/precip",
            "time_and_global_mean_bias/precip",
        ],
        "value": [-1, 2, 1, 0],
        "units": ["mm/day", "m", "mm/day", "mm/day"],
    }
)
MOVIE_URLS = {
    "baseline": ["gs://bucket/baseline/movie1.mp4", "gs://bucket/baseline/movie2.mp4"],
    "prognostic": ["gs://bucket/prognostic/movie1.mp4"],
}


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


def test_plot_2d_matplotlib():
    diagnostics = xarray.Dataset(
        {
            "a_somefilter": (
                ["x", "y"],
                np.arange(50).reshape((10, 5)),
                dict(long_name="longlongname", units="parsec/year"),
            ),
            "a_not": (["x", "y"], np.zeros((10, 5))),
        },
        attrs=dict(run="one-run"),
    )

    out = plot_2d_matplotlib(
        RunDiagnostics([diagnostics, diagnostics.assign_attrs(run="k")]),
        "somefilter",
        dims=["x", "y"],
        ylabel="y",
    )

    # make sure no errors are raised
    repr(out)


def test__get_metric_type_df():
    metrics = RunMetrics(METRICS_DF)
    table = _get_metric_type_df(metrics, "rmse_of_time_mean")
    expected_data = {"h500 [m]": [2, np.nan], "precip [mm/day]": [-1, 1]}
    expected_table = pd.DataFrame(expected_data, index=["run1", "run2"])
    pd.testing.assert_frame_equal(table, expected_table)


def test__get_metric_df():
    metrics = RunMetrics(METRICS_DF)
    table = _get_metric_df(metrics, {"rmse_of_time_mean": ["h500", "precip"]})
    expected_data = {
        "h500 rmse_of_time_mean [m]": [2, np.nan],
        "precip rmse_of_time_mean [mm/day]": [-1, 1],
    }
    expected_table = pd.DataFrame(expected_data, index=["run1", "run2"])
    pd.testing.assert_frame_equal(table, expected_table)


def test_get_movie_manifest():
    manifest = _get_movie_manifest(MOVIE_URLS, "gs://bucket/report")
    expected_manifest = [
        (
            "gs://bucket/baseline/movie1.mp4",
            "gs://bucket/report/_movies/baseline/movie1.mp4",
        ),
        (
            "gs://bucket/prognostic/movie1.mp4",
            "gs://bucket/report/_movies/prognostic/movie1.mp4",
        ),
        (
            "gs://bucket/baseline/movie2.mp4",
            "gs://bucket/report/_movies/baseline/movie2.mp4",
        ),
    ]
    assert set(manifest) == set(expected_manifest)


def test_get_public_links():
    links = _get_public_links(MOVIE_URLS, "gs://bucket/report")
    expected_links = {
        "movie1.mp4": [
            (
                "https://storage.googleapis.com/bucket/report/_movies/baseline/"
                "movie1.mp4",
                "baseline",
            ),
            (
                "https://storage.googleapis.com/bucket/report/_movies/prognostic/"
                "movie1.mp4",
                "prognostic",
            ),
        ],
        "movie2.mp4": [
            (
                "https://storage.googleapis.com/bucket/report/_movies/baseline/"
                "movie2.mp4",
                "baseline",
            )
        ],
    }
    assert links == expected_links


def test__get_cmap_kwargs():
    ds = xarray.Dataset(
        {
            "wind": (
                ["x", "y"],
                np.arange(50).reshape((10, 5)),
                dict(long_name="longlongname", units="parsec/year"),
            ),
        },
        attrs=dict(run="one-run"),
    )
    out = _get_cmap_kwargs(RunDiagnostics([ds, ds.assign_attrs(run="k")]), "wind")
    assert len(out) == 3
