import xarray
from runtime.monitor import Monitor
from runtime.names import DELP


def test_Monitor_monitor():
    ds = xarray.Dataset({"x": ([], 0.0), "y": (["z"], [1, 2]), DELP: (["z"], [1, 1])})
    tend = "tendency_of_x_due_to_blah"
    storage = "storage_of_y_path_due_to_blah"

    monitor = Monitor([tend, storage], ds, timestep=900)
    out = monitor("blah", lambda: {})()
    print(set(out))

    assert {tend, storage, "storage_of_mass_due_to_blah"} == set(out)
