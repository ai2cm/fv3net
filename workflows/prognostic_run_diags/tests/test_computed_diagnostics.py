import os

import fsspec
import numpy as np
import pytest
import xarray

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    ComputedDiagnosticsList,
    _parse_metadata,
    detect_rundirs,
    RunDiagnostics,
)


def test__parse_metadata():
    run = "blah-blah-baseline"
    out = _parse_metadata(run)
    assert out == {"run": run, "baseline": True}


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


def test_get_movie_links(tmpdir):
    domain = "http://www.domain.com"
    rdirs = ["rundir1", "rundir2"]
    for rdir in rdirs:
        rundir = tmpdir.mkdir(rdir)
        rundir.join("movie1.mp4").write("foobar")
        rundir.join("diags.nc").write("foobar")
        rundir.join("metrics.json").write("foobar")

    tmpdir.join(rdirs[0]).join("movie2.mp4").write("foobar")

    result = ComputedDiagnosticsList(str(tmpdir)).find_movie_links()

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
