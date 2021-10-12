import xarray
from runtime.monitor import Monitor
from runtime.names import DELP, TEMP


def test_monitor_inserts_tendency_and_storage_of_one_var():
    ds = xarray.Dataset(
        {
            "x": ([], 0.0),
            "y": (["z"], [1, 2]),
            DELP: (["z"], [1, 1]),
            TEMP: (["z"], [1, 1]),
        }
    )
    monitor = Monitor(
        tendency_variables={"x"},
        storage_variables={"y"},
        _state=ds,
        timestep=900,
        hydrostatic=True,
    )
    out = monitor("blah", lambda: {})()
    print(set(out))

    assert {
        "tendency_of_x_due_to_blah",
        "storage_of_y_path_due_to_blah",
        "storage_of_mass_due_to_blah",
        "storage_of_internal_energy_due_to_blah",
    } == set(out)


def test_Monitor_from_variables():
    variables = ["tendency_of_y_due_to_blah", "storage_of_z_path_due_to_yadayada"]

    monitor = Monitor.from_variables(
        variables, state={}, timestep=900, hydrostatic=False
    )
    assert {"y"} == monitor.tendency_variables
    assert {"z"} == monitor.storage_variables


def test_Monitor_checkpoint_returns_correct_variables():
    orig_value = xarray.DataArray(True)
    state = {
        "a": orig_value,
        "b": orig_value,
        "c": orig_value,
        "d": orig_value,
        DELP: orig_value,
        TEMP: orig_value,
    }

    monitor = Monitor(
        tendency_variables={"a", "b"},
        storage_variables={"c"},
        _state=state,
        timestep=900,
        hydrostatic=True,
    )

    immutable_state = monitor.checkpoint()
    assert {"a", "b", "c", DELP, TEMP} == set(immutable_state)


def test_Monitor_checkpoint_output_is_immutable():
    orig_value = xarray.DataArray(True)
    state = {"a": orig_value, DELP: orig_value, TEMP: orig_value}
    monitor = Monitor(
        tendency_variables={"a"},
        storage_variables=set(),
        _state=state,
        timestep=900,
        hydrostatic=True,
    )
    immutable_state = monitor.checkpoint()
    # mutate the state
    state["a"] = xarray.DataArray(False)
    # verify that the checkpoint is unaltered
    xarray.testing.assert_equal(orig_value, immutable_state["a"])
