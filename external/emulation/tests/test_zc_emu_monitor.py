from datetime import datetime, timedelta
import os
from typing import Mapping

import numpy as np
import pytest
import tensorflow as tf
import yaml
from emulation._monitor.monitor import (
    StorageHook,
    _convert_to_quantities,
    _convert_to_xr_dataset,
    _create_nc_path,
    _get_attrs,
    _remove_io_suffix,
)
from emulation._time import translate_time
from pace.util import Quantity
from xarray import DataArray


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


@pytest.mark.parametrize(
    "convert_func, expected_type",
    [(_convert_to_quantities, Quantity), (_convert_to_xr_dataset, DataArray)],
    ids=["to-quantity", "to-xr-dataset"],
)
def test__conversions(convert_func, expected_type):

    meta = dict(
        field_A={"units": "beepboop", "real-name": "missingno"},
        field_B={"units": "goobgob", "real-name": "trivial"},
    )

    state = {
        "field_A": np.ones(50).reshape(2, 25),
        "field_B": np.ones(50).reshape(2, 25),
        "singleton": np.array([1]),
    }

    quantities = convert_func(state, meta)

    assert quantities["field_A"].values.shape == (25, 2)
    assert quantities["field_B"].values.shape == (25, 2)
    assert quantities["singleton"].values.shape == ()

    for data in quantities.values():
        assert isinstance(data, expected_type)


def test__translate_time():

    year = 2016
    month = 10
    day = 29

    time = translate_time([year, month, day, None, 0, 0])

    assert time.year == year
    assert time.month == month
    assert time.day == day


def test__create_nc_path(dummy_rundir):

    _create_nc_path()
    assert os.path.exists(dummy_rundir / "netcdf_output")


def _get_data(save_nc, save_zarr, save_tfrecord=False):

    metadata_content = """
    air_temperature:
        units: K
    specific_humidity:
        units: kg/kg
    """

    config = StorageHook(
        output_freq_sec=900,
        metadata=yaml.safe_load(metadata_content),
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
        config.store(state)
        states.append(state)

    return states


def test_StorageHook_save_zarr(dummy_rundir):
    _get_data(save_nc=False, save_zarr=True)
    assert (dummy_rundir / "state_output.zarr").exists()


def test_StorageHook_save_nc(dummy_rundir):

    expected_states = _get_data(save_nc=True, save_zarr=False)
    nc_files = list((dummy_rundir / "netcdf_output").glob("*.nc"))
    assert len(nc_files) == len(expected_states)


def test_StorageHook_save_tf(dummy_rundir):
    expected_states = _get_data(False, False, True)
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


def test_StorageHook_does_not_modify_state():
    hook = StorageHook(output_freq_sec=1, save_nc=False, save_zarr=False)
    state = {"a": 0.0, "model_time": [2021, 1, 1, 0, 0, 0]}
    state_before = state.copy()
    hook.store(state)
    assert state == state_before


def test_StorageHook__store_data_at_time():

    hook = StorageHook(output_freq_sec=10, save_nc=False, save_zarr=False)
    t0 = datetime(2020, 1, 1, 0, 0, 0)
    hook.initial_time = t0
    assert hook._store_data_at_time(t0 + timedelta(seconds=10))

    hook = StorageHook(
        output_freq_sec=10, output_start_sec=11, save_nc=False, save_zarr=False
    )
    t0 = datetime(2020, 1, 1, 0, 0, 0)
    hook.initial_time = t0
    assert not hook._store_data_at_time(t0 + timedelta(seconds=10))
    assert hook._store_data_at_time(t0 + timedelta(seconds=20))

    hook = StorageHook(
        output_freq_sec=18000, output_start_sec=864_000, save_nc=False, save_zarr=False
    )
    hook.initial_time = datetime(2016, 3, 1, 0, 0, 0)
    assert hook._store_data_at_time(datetime(2016, 3, 11, 0, 0))
