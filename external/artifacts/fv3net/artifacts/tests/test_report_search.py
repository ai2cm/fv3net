import os
import pytest
from fv3net.artifacts.report_search import ReportIndex
from fv3net.artifacts.offline_report_search import _get_model_url

# flake8: noqa E501
REPORT_SAMPLE = """<html>
<head>
    <title>Prognostic run report</title>
    <script crossorigin="anonymous" integrity="sha384-T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js" type="text/javascript"></script>
    <script crossorigin="anonymous" integrity="sha384-98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js" type="text/javascript"></script>
    <script crossorigin="anonymous" integrity="sha384-" src="https://unpkg.com/@holoviz/panel@^0.10.3/dist/panel.min.js" type="text/javascript"></script>
</head>

<body>
    <h1>Prognostic run report</h1>

        Report created 2021-06-01 12:35:45 PDT


    <h2>Metadata</h2>
    <table>

        <tr>
            <th> verification dataset </th>
            <td> free_unperturbed_ssts_c384_fv3gfs_20170801_20190801 </td>
        </tr>

        <tr>
            <th> all-climate-ml-seed-0 </th>
            <td> gs://vcm-ml-experiments/spencerc/2021-05-28/n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-0/fv3gfs_run </td>
        </tr>

        <tr>
            <th> all-climate-ml-seed-1 </th>
            <td> gs://vcm-ml-experiments/spencerc/2021-05-28/n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-1/fv3gfs_run </td>
        </tr>

        <tr>
            <th> all-climate-ml-seed-2 </th>
            <td> gs://vcm-ml-experiments/spencerc/2021-05-28/n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-2/fv3gfs_run </td>
        </tr>

        <tr>
            <th> all-climate-ml-seed-3 </th>
            <td> gs://vcm-ml-experiments/spencerc/2021-05-28/n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-3/fv3gfs_run </td>
        </tr>

        <tr>
            <th> baseline </th>
            <td> gs://vcm-ml-experiments/spencerc/2021-05-24/n2f-25km-baseline-unperturbed/fv3gfs_run </td>
        </tr>

        <tr>
            <th> column_ML_wind_tendencies.mp4 </th>
            <td>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-0/column_ML_wind_tendencies.mp4'>all-climate-ml-seed-0</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-1/column_ML_wind_tendencies.mp4'>all-climate-ml-seed-1</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-2/column_ML_wind_tendencies.mp4'>all-climate-ml-seed-2</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-3/column_ML_wind_tendencies.mp4'>all-climate-ml-seed-3</a>
            </td>
        </tr>

        <tr>
            <th> column_heating_moistening.mp4 </th>
            <td>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-0/column_heating_moistening.mp4'>all-climate-ml-seed-0</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-1/column_heating_moistening.mp4'>all-climate-ml-seed-1</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-2/column_heating_moistening.mp4'>all-climate-ml-seed-2</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/all-climate-ml-seed-3/column_heating_moistening.mp4'>all-climate-ml-seed-3</a>
                <a href='https://storage.googleapis.com/vcm-ml-public/argo/prognostic-run-diags-xsl89/baseline/column_heating_moistening.mp4'>baseline</a>
            </td>
        </tr>

    </table>

    <h2>Links</h2>

    <a href="index.html">Home</a>

    <a href="hovmoller.html">Latitude versus time hovmoller</a>

    <a href="maps.html">Time-mean maps</a>

    <a href="zonal_pressure.html">Time-mean zonal-pressure profiles</a>

    <h2>Timeseries</h2>

    <div id='1184'>

        <div class="bk-root" id="b29040a2-d8a7-4665-88ad-5e2e46eacde1" data-root-id="1184"></div>
    </div>
    <script type="application/javascript>
"""

NEW_REPORT_SAMPLE = """<html>
    <head>
        <title>Prognostic run report</title>
        <script crossorigin="anonymous" integrity="sha384-T2yuo9Oe71Cz/I4X9Ac5+gpEa5a8PpJCDlqKYO0CfAuEszu1JrXLl8YugMqYe3sM" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.2.3.min.js" type="text/javascript"></script><script crossorigin="anonymous" integrity="sha384-98GDGJ0kOMCUMUePhksaQ/GYgB3+NH9h996V88sh3aOiUNX3N+fLXAtry6xctSZ6" src="https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.3.min.js" type="text/javascript"></script><script crossorigin="anonymous" integrity="sha384-" src="https://unpkg.com/@holoviz/panel@^0.10.3/dist/panel.min.js" type="text/javascript"></script>
    </head>

    <body>
    <h1>Prognostic run report</h1>
    Report created 2021-09-13 09:50:30 PDT

        <h2>Metadata</h2>
<pre id="json">
{
    "verification dataset": "40day_may2020",
    "all-climate-ml-seed-0": "gs://vcm-ml-experiments/spencerc/2021-05-28/n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-0/fv3gfs_run",
    "n2f-3hr-prescribe-Q1-Q2": "gs://vcm-ml-experiments/default/2021-09-10/n2f-3km-prescribe-q1-q2-3hr/fv3gfs_run",
    "n2f-24hr-prescribe-Q1-Q2": "gs://vcm-ml-experiments/default/2021-09-10/n2f-3km-prescribe-q1-q2-24hr/fv3gfs_run",
    "column_heating_moistening.mp4": "<a href='https://storage.googleapis.com/vcm-ml-public/argo/2021-09-13t094629-2733d02e6fb8/_movies/n2f-3hr-control/column_heating_moistening.mp4'>n2f-3hr-control</a> <a href='https://storage.googleapis.com/vcm-ml-public/argo/2021-09-13t094629-2733d02e6fb8/_movies/n2f-3hr-prescribe-Q1-Q2/column_heating_moistening.mp4'>n2f-3hr-prescribe-Q1-Q2</a> <a href='https://storage.googleapis.com/vcm-ml-public/argo/2021-09-13t094629-2733d02e6fb8/_movies/n2f-24hr-prescribe-Q1-Q2/column_heating_moistening.mp4'>n2f-24hr-prescribe-Q1-Q2</a>"
}
</pre>

        <h2>Links</h2>

                <ol>
<li><a href="index.html">Home</a></li>
<li><a href="process_diagnostics.html">Process diagnostics</a></li>
<li><a href="hovmoller.html">Latitude versus time hovmoller</a></li>
<li><a href="maps.html">Time-mean maps</a></li>
<li><a href="zonal_pressure.html">Time-mean zonal-pressure profiles</a></li>
</ol>


        <h2>Top-level metrics</h2>
"""

OFFLINE_REPORT_SAMPLE = """
    <html>
    <head>
        <title>ML offline diagnostics</title>

    </head>

    <body>
    <h1>ML offline diagnostics</h1>
    Report created 2021-11-07 01:02:49 PDT

        <h2>Metadata</h2>
<pre id="json">
{
    "model_path": "gs://vcm-ml-experiments/default/2021-11-07/hybrid-fluxes-input-3hrly-rf/trained_models/sklearn_model",
    "data_config": {
        "mapper_config": {
            "function": "open_3hrly_fine_resolution_nudging_hybrid",
            "kwargs": {
                "fine_url": "gs://vcm-ml-intermediate/2021-10-08-fine-res-3hrly-averaged-Q1-Q2-from-40-day-X-SHiELD-simulation-2020-05-27.zarr",
                "nudge_url": "gs://vcm-ml-experiments/default/2021-09-26/n2f-3km-nudging/fv3gfs_run"
            }
        },
        "function": "batches_from_mapper",
        "kwargs": {
            "timesteps_per_batch": 10,
            "timesteps": [
                "20160902.203000",
                "20160908.203000",
                "20160906.083000",
                "20160904.023000",
                "20160904.233000",
                "20160901.083000",
                "20160905.083000",
                "20160909.233000",
                "20160904.143000",
                "20160903.233000",
                "20160907.083000",
                "20160902.083000",
                "20160903.173000",
                "20160904.113000",
                "20160907.143000",
                "20160901.113000",
                "20160901.203000",
"""


@pytest.mark.parametrize("report_sample", (REPORT_SAMPLE, NEW_REPORT_SAMPLE))
def test_ReportIndex_compute(report_sample, tmpdir):
    expected_run = (
        "gs://vcm-ml-experiments/spencerc/2021-05-28/"
        "n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-0/fv3gfs_run"
    )
    os.makedirs(tmpdir.join("report1"))
    os.makedirs(tmpdir.join("report2"))
    with open(tmpdir.join("report1/index.html"), "w") as f:
        f.write(report_sample)
    with open(tmpdir.join("report2/index.html"), "w") as f:
        f.write(report_sample)

    index = ReportIndex()
    index.compute(str(tmpdir))

    assert expected_run in index.reports_by_id
    assert set(index.reports_by_id[expected_run]) == {
        str(tmpdir.join("report1/index.html")),
        str(tmpdir.join("report2/index.html")),
    }


def test_ReportIndex_compute_with_recurse_once(tmpdir):
    expected_run = (
        "gs://vcm-ml-experiments/spencerc/2021-05-28/"
        "n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-0/fv3gfs_run"
    )
    os.makedirs(tmpdir.join("topdir/report1"))
    os.makedirs(tmpdir.join("topdir/report2"))
    with open(tmpdir.join("topdir/report1/index.html"), "w") as f:
        f.write(NEW_REPORT_SAMPLE)
    with open(tmpdir.join("topdir/report2/index.html"), "w") as f:
        f.write(NEW_REPORT_SAMPLE)

    index = ReportIndex()
    index.compute(str(tmpdir), recurse_once=True)
    assert expected_run in index.reports_by_id


def test_ReportIndex_public_links():
    index = ReportIndex(
        {
            "/path/to/run": [
                "gs://bucket/report1/index.html",
                "gs://bucket/report2/index.html",
            ]
        },
    )
    expected_public_links = {
        "https://storage.googleapis.com/bucket/report1/index.html",
        "https://storage.googleapis.com/bucket/report2/index.html",
    }
    assert set(index.public_links("/path/to/run")) == expected_public_links
    assert len(index.public_links("/run/not/in/index")) == 0


def test_ReportIndex_reports():
    index = ReportIndex(
        {
            "/path/to/run": ["/report1/index.html", "/report2/index.html"],
            "/path/to/other/run": ["/report1/index.html"],
        },
    )
    expected_reports = {"/report1/index.html", "/report2/index.html"}
    assert index.reports == expected_reports


def test_offline_report_search(tmpdir):
    expected_model_path = (
        "gs://vcm-ml-experiments/default/2021-11-07/"
        "hybrid-fluxes-input-3hrly-rf/trained_models/sklearn_model"
    )
    os.makedirs(tmpdir.join("topdir/report1"))
    with open(tmpdir.join("topdir/report1/index.html"), "w") as f:
        f.write(OFFLINE_REPORT_SAMPLE)

    index = ReportIndex(_search_function=_get_model_url)
    index.compute(str(tmpdir), recurse_once=True)
    assert expected_model_path in index.reports_by_id
