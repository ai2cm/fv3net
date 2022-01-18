import os
from typing import Mapping

import numpy as np
import pytest
import tensorflow as tf
from emulation.hooks.monitor import (
    StorageHook,
    _bool_from_str,
    _convert_to_quantities,
    _convert_to_xr_dataset,
    _create_nc_path,
    _get_attrs,
    _get_timestep,
    _load_nml,
    _remove_io_suffix,
    _translate_time,
)
from fv3gfs.util import Quantity
from xarray import DataArray


def save_var_metadata(path):
    metadata_content = """
    air_temperature:
        units: K
    specific_humidity:
        units: kg/kg
    """

    with open(path, "w") as f:
        f.write(metadata_content)


@pytest.mark.parametrize(
    "value, expected",
    [
        ("y", True),
        ("yes", True),
        ("true", True),
        ("n", False),
        ("no", False),
        ("false", False),
    ],
)
def test__bool_from_str(value, expected):

    assert _bool_from_str(value) == expected
    assert _bool_from_str(value.capitalize()) == expected
    assert _bool_from_str(value.upper()) == expected


def test__bool_from_str_unrecognized():

    with pytest.raises(ValueError):
        _bool_from_str("not-translatable-to-bool")


def test__load_nml(dummy_rundir):

    namelist = _load_nml()
    assert namelist["coupler_nml"]["hours"] == 1


def test__get_timestep(dummy_rundir):
    namelist = _load_nml()
    timestep = _get_timestep(namelist)

    assert timestep == 900


@pytest.mark.parametrize(
    "value,expected",
    [
        ("a_input", "a"),
        ("a_output", "a"),
        ("a_input_a", "a_input_a"),
        ("a_output_output", "a_output"),
    ],
    ids=["input-remove", "no-change", "output-remove", "double-match-no-change"],
)
def test__remove_io_suffix(value, expected):

    result = _remove_io_suffix(value)
    assert result == expected


def test__get_attrs():

    metadata = dict(number={"a_number": 1, "a_list": [1, 2, 3]})

    dat = _get_attrs("number", metadata)
    for k, v in dat.items():
        assert isinstance(v, str)

    dat = _get_attrs("nonexistent", metadata)
    assert not dat
    assert isinstance(dat, Mapping)


def get_state_meta_for_conversion():

    metadata = dict(
        field_A={"units": "beepboop", "real-name": "missingno"},
        field_B={"units": "goobgob", "real-name": "trivial"},
    )

    state = {
        "field_A": np.ones(50).reshape(2, 25),
        "field_B": np.ones(50).reshape(2, 25),
    }

    return state, metadata


@pytest.mark.parametrize(
    "convert_func, expected_type",
    [(_convert_to_quantities, Quantity), (_convert_to_xr_dataset, DataArray)],
    ids=["to-quantity", "to-xr-dataset"],
)
def test__conversions(convert_func, expected_type):

    state, meta = get_state_meta_for_conversion()
    quantities = convert_func(state, meta)

    for data in quantities.values():
        assert isinstance(data, expected_type)
        assert data.values.shape == (25, 2)


def test__translate_time():

    year = 2016
    month = 10
    day = 29

    time = _translate_time([year, month, day, None, 0, 0])

    assert time.year == year
    assert time.month == month
    assert time.day == day


def test__create_nc_path(dummy_rundir):

    _create_nc_path()
    assert os.path.exists(dummy_rundir / "netcdf_output")


def _get_data(dummy_rundir, save_nc, save_zarr, save_tfrecord=False):

    meta_path = dummy_rundir / "var_metadata.yaml"
    save_var_metadata(meta_path)
    config = StorageHook(
        meta_path,
        900,
        save_nc=save_nc,
        save_zarr=save_zarr,
        save_tfrecord=save_tfrecord,
    )
    n = 7
    batch = 10

    states = []

    for i in range(1, n + 1):
        state = {
            "model_time": [2016, 10, i, 0, 0, 0],
            # model returns data in fortran (row-major) order (z, sample)
            "air_temperature": np.arange(790).reshape(79, batch),
            "specific_humidity": np.arange(790).reshape(79, batch),
        }
        config(state)
        states.append(state)

    return dummy_rundir, states


def test_StorageHook_save_zarr(dummy_rundir):
    _get_data(dummy_rundir, save_nc=False, save_zarr=True)
    assert (dummy_rundir / "state_output.zarr").exists()


def test_StorageHook_save_nc(dummy_rundir):

    dummy_rundir, expected_states = _get_data(
        dummy_rundir, save_nc=True, save_zarr=False
    )
    nc_files = list((dummy_rundir / "netcdf_output").glob("*.nc"))
    assert len(nc_files) == len(expected_states)


def test_StorageHook_save_tf(dummy_rundir):
    dummy_rundir, expected_states = _get_data(dummy_rundir, False, False, True)
    tf_records_path = dummy_rundir / "tfrecords"

    # load tf records
    # a little boiler plate
    parser = tf.saved_model.load(str(tf_records_path / "parser.tf"))
    tf_ds = tf.data.TFRecordDataset(
        tf.data.Dataset.list_files(str(tf_records_path / "*.tfrecord"))
    ).map(parser.parse_single_example)

    # assertions
    state = expected_states[0]
    n = len(expected_states)
    state["time"] = np.array(["time"] * n)
    for key in tf_ds.element_spec:
        if len(state[key].shape):
            assert (
                tuple(tf_ds.element_spec[key].shape[1:]) == state[key].T.shape[1:]
            ), key
            assert tf_ds.element_spec[key].shape[0] is None
    assert len(list(tf_ds)) == n


def test_error_on_call():

    with pytest.raises(ValueError):
        from emulation import store

        store({})
