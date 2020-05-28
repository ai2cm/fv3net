import pytest
import os
import xarray as xr
import synth

from synth import generate_restart_data


def test_generate_restart_data():
    output = generate_restart_data(n=48, n_soil=4, nz=79)
    assert isinstance(output["fv_core.res"][1], xr.Dataset)


def test_generate_restart_data_contains_time():
    output = generate_restart_data(n=48, n_soil=4, nz=79)
    assert "Time" in output["fv_core.res"][1].coords
