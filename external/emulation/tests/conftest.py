import contextlib
import os
import pytest


@contextlib.contextmanager
def set_current_directory(path):

    origin = os.getcwd()

    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def save_input_nml(path):

    input_nml_content = """
    &coupler_nml
        calendar = 'julian'
        current_date = 2016, 8, 1, 0, 0, 0
        dt_atmos = 900
        days = 0
        hours = 1
        minutes = 0
        months = 0
        seconds = 0
    /

    &fv_core_nml
        layout = 1,1
    /
    """

    with open(path, "w") as f:
        f.write(input_nml_content)


@pytest.fixture
def dummy_rundir(tmp_path):

    rundir = tmp_path
    with set_current_directory(rundir):
        save_input_nml(rundir / "input.nml")

        yield rundir
