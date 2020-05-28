import pytest
import os
import xarray as xr
import synth

from synth import generate_restart_data


@pytest.fixture()
def output():
    return generate_restart_data(n=48, n_soil=4, nz=79)


def test_generate_restart_data(output):
    assert isinstance(output["fv_core.res"][1], xr.Dataset)


def test_generate_restart_data_contains_time(output):
    assert "Time" in output["fv_core.res"][1].coords


def test_generate_restart_data_all_tiles_present(output):
    for _, val in output.items():
        assert set(val.keys()) == {1, 2, 3, 4, 5, 6}


def test_generate_restart_data_all_keys_present(output):
    keys = {"fv_core.res", "fv_tracer.res", "fv_srf_wnd.res", "sfc_data"}
    assert set(output) == keys
