import xarray
from runtime.monitor import Monitor
from runtime.names import DELP


def test_monitor_inserts_tendency_and_storage_of_one_var():
    ds = xarray.Dataset({"x": ([], 0.0), "y": (["z"], [1, 2]), DELP: (["z"], [1, 1])})
    monitor = Monitor(
        tendency_variables={"x"}, storage_variables={"y"}, _state=ds, timestep=900
    )
    out = monitor("blah", lambda: {})()
    print(set(out))

    assert {
        "tendency_of_x_due_to_blah",
        "storage_of_y_path_due_to_blah",
        "storage_of_mass_due_to_blah",
    } == set(out)


def test_Monitor_from_variables():
    variables = ["tendency_of_y_due_to_blah", "storage_of_z_path_due_to_yadayada"]

    monitor = Monitor.from_variables(variables, state={}, timestep=900)
    assert {"y"} == monitor.tendency_variables
    assert {"z"} == monitor.storage_variables
