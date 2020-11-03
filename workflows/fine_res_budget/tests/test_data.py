from budget.data import open_atmos_avg


def test_open_atmos_avg():
    dataset = open_atmos_avg()
    assert set(dataset.dims) <= {
        "nv",
        "grid_xt",
        "grid_y",
        "time",
        "grid_yt",
        "tile",
        "grid_x",
        "pfull",
    }
