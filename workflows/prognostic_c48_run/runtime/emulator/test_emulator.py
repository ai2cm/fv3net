import argparse
import os
import datetime
import numpy as np
import tensorflow as tf
import xarray as xr
from runtime.emulator.batch import to_dict
from runtime.emulator.emulator import Config as OnlineEmulatorConfig, OnlineEmulator
from runtime.emulator.adapter import (
    get_xarray_emulator,
    update_state_with_emulator,
    _update_state_with_emulator,
    XarrayEmulator,
)

from runtime.emulator.loss import MultiVariableLoss, RHLoss, QVLoss
from .utils import _get_argsin
import pytest
from hypothesis import given
from hypothesis.strategies import lists, integers


@pytest.fixture(scope="function")
def change_test_dir(tmpdir, request):
    # from https://stackoverflow.com/a/62055409/1208392
    os.chdir(tmpdir)
    yield
    os.chdir(request.config.invocation_dir)


def test_OnlineEmulator_partial_fit(state):
    config = OnlineEmulatorConfig(
        batch_size=32, learning_rate=0.001, momentum=0.0, levels=63
    )

    emulator = get_xarray_emulator(config)
    emulator.partial_fit(state, state)


def test_OnlineEmulator_partial_fit_logged(state):
    config = OnlineEmulatorConfig(
        batch_size=8, learning_rate=0.0001, momentum=0.0, levels=63
    )
    time = datetime.datetime.now().isoformat()

    emulator = get_xarray_emulator(config)
    writer = tf.summary.create_file_writer(f"logs_long/{time}")
    with writer.as_default():
        for i in range(10):
            emulator.partial_fit(state, state)


def test_OnlineEmulator_fails_when_accessing_nonexistant_var(state):
    config = OnlineEmulatorConfig(
        batch_size=32,
        learning_rate=0.001,
        momentum=0.0,
        extra_input_variables=["not a varialbe in any state 332r23r90e9d"],
        levels=63,
    )

    emulator = get_xarray_emulator(config)
    with pytest.raises(KeyError):
        emulator.partial_fit(state, state)


@pytest.mark.parametrize(
    "extra_inputs", [[], ["cos_zenith_angle", "surface_pressure"]],
)
def test_OnlineEmulator_fit_predict(state, extra_inputs):
    config = OnlineEmulatorConfig(
        batch_size=32,
        learning_rate=0.001,
        momentum=0.0,
        extra_input_variables=extra_inputs,
        levels=63,
    )

    emulator = get_xarray_emulator(config)
    emulator.partial_fit(state, state)
    stateout = emulator.predict(state)
    assert isinstance(stateout, dict)
    assert list(stateout["eastward_wind"].dims) == ["z", "y", "x"]


@pytest.mark.parametrize("with_validation", [True, False])
@pytest.mark.parametrize(
    "config",
    [
        OnlineEmulatorConfig(
            batch_size=32, learning_rate=0.001, momentum=0.0, levels=79
        ),
        OnlineEmulatorConfig(
            batch_size=32,
            learning_rate=0.001,
            momentum=0.0,
            target=QVLoss(0),
            levels=79,
        ),
        OnlineEmulatorConfig(target=RHLoss(50), levels=79,),
    ],
)
def test_OnlineEmulator_batch_fit(config, with_validation):
    x = to_dict(_get_argsin(config.levels))
    dataset = tf.data.Dataset.from_tensors((x, x)).unbatch()

    emulator = OnlineEmulator(config)

    if with_validation:
        emulator.batch_fit(dataset, validation_data=dataset)
    else:
        emulator.batch_fit(dataset)


def test_top_level():
    dict_ = {"target": {"level": 10}}
    config = OnlineEmulatorConfig.from_dict(dict_)
    assert QVLoss(10) == config.target


@pytest.mark.parametrize("output_exists", [True, False])
def test_dump_load_OnlineEmulator(state, tmpdir, output_exists):
    if output_exists:
        path = str(tmpdir)
    else:
        path = str(tmpdir.join("model"))

    n = state["air_temperature"].sizes["z"]
    config = OnlineEmulatorConfig(levels=n)
    emulator = get_xarray_emulator(config)
    emulator.partial_fit(state, state)
    emulator.dump(path)
    new_emulator = XarrayEmulator.load(path)

    # assert that the air_temperature output is unchanged
    field = "air_temperature"
    np.testing.assert_array_equal(
        new_emulator.predict(state)[field], emulator.predict(state)[field]
    )


@pytest.mark.parametrize(
    "args,loss_cls",
    [
        (["--level", "50"], QVLoss),
        (["--level", "50", "--relative-humidity"], RHLoss),
        (["--multi-output"], MultiVariableLoss),
        (["--multi-output", "--relative-humidity"], MultiVariableLoss),
    ],
)
def test_OnlineEmulatorConfig_register_parser(args, loss_cls):
    parser = argparse.ArgumentParser()
    OnlineEmulatorConfig.register_parser(parser)
    args = parser.parse_args(args)
    config = OnlineEmulatorConfig.from_args(args)
    assert isinstance(config, OnlineEmulatorConfig)
    assert isinstance(config.target, loss_cls)

    if args.relative_humidity:
        assert config.relative_humidity
    else:
        assert not config.relative_humidity


@given(lists(integers(min_value=0)))
def test_OnlineEmulatorConfig_multi_output_levels(levels):

    str_levels = ",".join(str(s) for s in levels)

    parser = argparse.ArgumentParser()
    OnlineEmulatorConfig.register_parser(parser)
    args = parser.parse_args(["--multi-output", "--levels", str_levels])
    config = OnlineEmulatorConfig.from_args(args)

    assert config.target.levels == levels


@pytest.mark.network
def test_checkpointed_model(tmpdir):

    # dump a model
    config = OnlineEmulatorConfig()
    emulator = OnlineEmulator(config)
    emulator.dump(tmpdir)

    # load it
    emulator = OnlineEmulator.load(str(tmpdir))
    assert isinstance(emulator, OnlineEmulator)


def test_update_state_with_emulator(state):
    qv = "specific_humidity"
    old_state = state
    state = state.copy()

    z_slice = slice(0, 10)
    new = {qv: state[qv] + 1.0}

    _update_state_with_emulator(
        state, new, from_orig=lambda name, arr: xr.DataArray(name == qv) & (arr.z < 10)
    )

    xr.testing.assert_allclose(state[qv].sel(z=z_slice), old_state[qv].sel(z=z_slice))
    new_z_slice = slice(10, None)
    xr.testing.assert_allclose(state[qv].sel(z=new_z_slice), new[qv].sel(z=new_z_slice))


@pytest.mark.parametrize("ignore_humidity_below", [0, 65, None])
def test_integration_with_from_orig(state, ignore_humidity_below):
    update_state_with_emulator(
        state, state, ignore_humidity_below=ignore_humidity_below
    )
