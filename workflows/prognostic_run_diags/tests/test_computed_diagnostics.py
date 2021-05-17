import os

import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    ComputedDiagnosticsList,
    _parse_metadata,
    detect_folders,
    RunDiagnostics,
    RunMetrics,
    DiagnosticFolder,
)


def test__parse_metadata():
    run = "blah-blah-baseline"
    out = _parse_metadata(run)
    assert out == {"run": run, "baseline": True}


def test_detect_folders(tmpdir):

    fs = fsspec.filesystem("file")

    rundirs = ["rundir1", "rundir2"]
    for rdir in rundirs:
        tmpdir.mkdir(rdir).join("diags.nc").write("foobar")

    tmpdir.mkdir("not_a_rundir").join("useless_file.txt").write("useless!")

    result = detect_folders(tmpdir, fs)

    assert len(result) == 2
    for found_dir in result:
        assert found_dir in rundirs
        assert isinstance(result[found_dir], DiagnosticFolder)


def test_ComputedDiagnosticsList_from_urls():
    rundirs = ["rundir1", "rundir2"]
    result = ComputedDiagnosticsList.from_urls(rundirs)

    assert len(result.folders) == 2
    assert isinstance(result.folders["0"], DiagnosticFolder)
    assert isinstance(result.folders["1"], DiagnosticFolder)


def test_detect_folders_fail_less_than_2(tmpdir):

    fs = fsspec.filesystem("file")

    tmpdir.mkdir("rundir1").join("diags.nc").write("foobar")

    with pytest.raises(ValueError):
        detect_folders(tmpdir, fs)


def test_get_movie_links(tmpdir):
    domain = "http://www.domain.com"
    rdirs = ["rundir1", "rundir2"]
    for rdir in rdirs:
        rundir = tmpdir.mkdir(rdir)
        rundir.join("movie1.mp4").write("foobar")
        rundir.join("diags.nc").write("foobar")
        rundir.join("metrics.json").write("foobar")

    tmpdir.join(rdirs[0]).join("movie2.mp4").write("foobar")

    result = ComputedDiagnosticsList.from_directory(str(tmpdir)).find_movie_links()

    assert "movie1.mp4" in result
    assert "movie2.mp4" in result
    assert {(os.path.join(domain, tmpdir, "rundir1", "movie2.mp4"), "rundir1")} == set(
        result["movie2.mp4"]
    )


one_run = xarray.Dataset({"a": ([], 1,), "b": ([], 2)}, attrs=dict(run="one-run"))


@pytest.mark.parametrize("varname", ["a", "b"])
@pytest.mark.parametrize("run", ["one-run", "another-run"])
def test_RunDiagnostics_get_variable(run, varname):
    another_run = one_run.assign_attrs(run="another-run")
    arr = RunDiagnostics([one_run, another_run]).get_variable(run, varname)
    xarray.testing.assert_equal(arr, one_run[varname])


def test_RunDiagnostics_get_variable_missing_variable():
    another_run = one_run.assign_attrs(run="another-run").drop("a")
    run = "another-run"
    varname = "a"

    arr = RunDiagnostics([one_run, another_run]).get_variable(run, varname)
    xarray.testing.assert_equal(arr, xarray.full_like(one_run[varname], np.nan))


def test_RunDiagnostics_runs():
    runs = ["a", "b", "c"]
    diagnostics = RunDiagnostics([one_run.assign_attrs(run=run) for run in runs])
    assert diagnostics.runs == runs


def test_RunDiagnostics_list_variables():
    ds = xarray.Dataset({})
    diagnostics = RunDiagnostics(
        [
            ds.assign(a=1, b=1).assign_attrs(run="1"),
            ds.assign(b=1).assign_attrs(run="2"),
            ds.assign(c=1).assign_attrs(run="3"),
        ]
    )
    assert diagnostics.variables == {"a", "b", "c"}


metrics_df = pd.DataFrame(
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


def test_RunMetrics_runs():
    metrics = RunMetrics(metrics_df)
    assert metrics.runs == ["run1", "run2"]


def test_RunMetrics_types():
    metrics = RunMetrics(metrics_df)
    assert metrics.types == {"rmse_of_time_mean", "time_and_global_mean_bias"}


def test_RunMetrics_get_metric_variables():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_variables("rmse_of_time_mean") == {"precip", "h500"}


def test_RunMetrics_get_metric_value():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_value("rmse_of_time_mean", "precip", "run1") == -1


def test_RunMetrics_get_metric_value_missing():
    metrics = RunMetrics(metrics_df)
    assert np.isnan(metrics.get_metric_value("rmse_of_time_mean", "h500", "run2"))


def test_RunMetrics_get_metric_units():
    metrics = RunMetrics(metrics_df)
    assert metrics.get_metric_units("rmse_of_time_mean", "precip", "run1") == "mm/day"


@pytest.fixture()
def url():
    # this data might get out of date if it does we should replace it with synth
    # data
    return "gs://vcm-ml-public/argo/2021-05-05-compare-n2o-resolution-a8290d64ce4f/"


@pytest.mark.network
def test_ComputeDiagnosticsList_load_diagnostics(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    meta, diags = diags.load_diagnostics()
    assert isinstance(diags, RunDiagnostics)


@pytest.mark.network
def test_ComputeDiagnosticsList_load_metrics(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    meta = diags.load_metrics()
    assert isinstance(meta, RunMetrics)


@pytest.mark.network
def test_ComputeDiagnosticsList_find_movie_links(url):
    diags = ComputedDiagnosticsList.from_directory(url)
    meta = diags.find_movie_links()
    assert len(meta) == 0
