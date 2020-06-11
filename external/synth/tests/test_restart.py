import pytest
import xarray as xr
import numpy as np

from synth import generate_restart_data


@pytest.fixture()
def output():
    return generate_restart_data(nx=48, n_soil=4, nz=79)


def test_generate_restart_data(output):
    assert isinstance(output["fv_core.res"][1], xr.Dataset)


def test_generate_restart_data_contains_time(output):
    assert "Time" in output["fv_core.res"][1].coords


def test_generate_restart_data_all_tiles_present(output):
    for _, val in output.items():
        assert set(val.keys()) == set(range(1, 7))


def test_generate_restart_data_all_keys_present(output):
    keys = {"fv_core.res", "fv_tracer.res", "fv_srf_wnd.res", "sfc_data"}
    assert set(output) == keys


def _yield_all_variables_and_coords(output):

    # use stack to recurse into the "output" data-structure
    # could implement with recursion as well
    stack = [output]
    while len(stack) > 0:
        item = stack.pop()
        try:
            # item is a dataset
            for variable in list(item.data_vars) + list(item.coords):
                yield variable, item[variable]
        except AttributeError:
            # item is a dict
            stack.extend(item.values())


def test_all_data_float32(output):
    for variable, array in _yield_all_variables_and_coords(output):
        assert array.dtype == np.float32, variable


def test_u_correct(output):
    assert output["fv_core.res"][1].u.dims == ("Time", "zaxis_1", "yaxis_1", "xaxis_1")


@pytest.mark.parametrize("include_agrid_winds", [False, True])
def test_include_agrid_winds_option(include_agrid_winds):
    fv_core = generate_restart_data(
        nx=48, n_soil=4, nz=79, include_agrid_winds=include_agrid_winds
    )["fv_core.res"][1]
    assert include_agrid_winds == ("ua" in fv_core and "va" in fv_core)
