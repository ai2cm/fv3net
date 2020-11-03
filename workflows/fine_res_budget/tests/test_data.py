import xarray as xr
from budget.data import open_atmos_avg, open_merged


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


def test_open_merged(data_dirs):
    diags_url, restarts_url = data_dirs[:2]
    dataset = open_merged(restarts_url, diags_url)
    assert isinstance(dataset, xr.Dataset)
