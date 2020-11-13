import pytest
import numpy as np

from emulation_slim import packer


def test_feature_size_to_slices():

    sizes = {
        "var1": 1,
        "var2": 10,
        "var3": 4
    }

    expected_slices = {
        "var1": slice(0, 1),
        "var2": slice(1, 11),
        "var3": slice(11, 15)
    }

    result = packer.feature_size_to_slices(sizes)

    assert len(result) == len(expected_slices)
    for varname, vslice in expected_slices.items():
        assert varname in result
        assert expected_slices[varname] == result[varname]


@pytest.fixture
def arr_2d():
    return np.ones((3, 5))

@pytest.fixture
def arr_3d():
    return np.arange(30).reshape(2, 3, 5)


def test__split_on_tracer_dim(arr_3d):

    result = packer._split_on_tracer_dim("var", arr_3d)

    for i, varname in enumerate(["var_0", "var_1"]):
        assert varname in result
        np.testing.assert_equal(arr_3d[i], result[varname])


def test_split_tracer_fields(arr_2d, arr_3d):

    state = {
        "var_2d": arr_2d,
        "var_3d": arr_3d
    }

    packer.split_tracer_fields(state)

    assert len(state) == 3
    assert "var_2d" in state
    assert "var_3d" not in state
    for varname in ["var_3d_0", "var_3d_1"]:
        assert varname in state

def test_split_tracer_fields_no_tracers():

    arr_2d = np.ones((3, 5))

    state = {
        "var_0": arr_2d,
        "var_1": arr_2d,
    }

    packer.split_tracer_fields(state)

    assert len(state) == 2
    for varname in ["var_0", "var_1"]:
        assert varname in state
        np.testing.assert_equal(state[varname], arr_2d)


def test__detect_tracer_field_names():

    state = {
        "var1_input": None,
        "var1_output": None,
        "var2_input_0": None,
        "var2_input_1": None,
        "var2_input_2": None,
        "var3_output_0": None,
        "var3_output_1": None,
    }

    tracer_info = packer._detect_tracer_field_names(state)
    assert len(tracer_info) == 2
    assert "var2_input" in tracer_info
    assert tracer_info["var2_input"] == ["var2_input_0", "var2_input_1", "var2_input_2"]
    assert "var3_output" in tracer_info
    assert tracer_info["var3_output"] == ["var3_output_0", "var3_output_1"]


def test__detect_tracer_field_names_noncontiguous():

    state = {
        "var1_input": None,
        "var1_output": None,
        "var2_input_0": None,
        "var2_input_1": None,
        "var2_input_3": None,
    }

    with pytest.raises(KeyError):
        _ = packer._detect_tracer_field_names(state)


def test_consolidate_tracers(arr_2d, arr_3d):

    state = {
        "var1_input": arr_2d,
        "var1_output": arr_2d,
        "var2_input_0": arr_3d[0],
        "var2_input_1": arr_3d[1],
        "var3_output_0": arr_3d[0],
        "var3_output_1": arr_3d[1],
    }

    tracer_info = {
        "var2_input": ["var2_input_0", "var2_input_1"],
        "var3_output": ["var3_output_0", "var3_output_1"]
    }

    packer.consolidate_tracers(state, tracer_info)

    np.testing.assert_equal(state["var2_input"], arr_3d)
    np.testing.assert_equal(state["var3_output"], arr_3d)


def test_EmuArrayPacker(arr_2d, arr_3d):

    arr_packer = packer.EmuArrayPacker(
        ["var0_input", "var1_input"],
        {"var0_input": arr_2d.shape[0], "var1_input": arr_2d.shape[0]},
        {}
    )

    state = {
        "var0_input": arr_2d,
        "var1_input": arr_2d
    }

    packed = arr_packer.to_array(state)
    np.testing.assert_equal(packed, np.concatenate([arr_2d.T, arr_2d.T], axis=1))
    unpacked = arr_packer.to_dict(packed)

    for varname, expected in state.items():
        assert varname in unpacked
        np.testing.assert_equal(expected, unpacked[varname])
