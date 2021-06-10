import os
import fsspec
from fv3net.artifacts.report_search import ReportIndex

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


def test_ReportIndex_compute(tmpdir):
    expected_run = (
        "gs://vcm-ml-experiments/spencerc/2021-05-28/"
        "n2f-25km-all-climate-tapered-fixed-ml-unperturbed-seed-0/fv3gfs_run"
    )
    os.makedirs(tmpdir.join("report1"))
    os.makedirs(tmpdir.join("report2"))
    with open(tmpdir.join("report1/index.html"), "w") as f:
        f.write(REPORT_SAMPLE)
    with open(tmpdir.join("report2/index.html"), "w") as f:
        f.write(REPORT_SAMPLE)

    index = ReportIndex()
    index.compute(str(tmpdir))

    assert expected_run in index.reports_by_run
    assert set(index.reports_by_run[expected_run]) == {
        str(tmpdir.join("report1/index.html")),
        str(tmpdir.join("report2/index.html")),
    }


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
