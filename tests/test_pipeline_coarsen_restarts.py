import pytest
import os
import xarray as xr
import synth

from fv3net.pipelines.coarsen_restarts.testing import (
    fv_core_schema,
    generate_restart_data,
)


def test_fv_core_schema(regtest):
    print(fv_core_schema(48, 79), file=regtest)


def test_generate_restart_data():
    output = generate_restart_data(n=48, n_soil=4, nz=79)
    assert isinstance(output["fv_core.res"][1], xr.Dataset)


def test_generate_restart_data_contains_time():
    output = generate_restart_data(n=48, n_soil=4, nz=79)
    assert "Time" in output["fv_core.res"][1].coords


def test_generate_fv_core_integration():
    schema = fv_core_schema(n=2, nz=3)
    synth.generate(schema)
