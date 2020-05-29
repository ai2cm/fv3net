import pytest
import xarray as xr
import numpy as np

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
