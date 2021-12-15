import xarray as xr
from budget.data import open_merged
import budget.config


def test_open_merged(data_dirs):
    diags_url, restarts_url, gfsphysics_url, area_url = data_dirs
    dataset = open_merged(restarts_url, diags_url, gfsphysics_url, area_url)
    assert isinstance(dataset, xr.Dataset)

    for name in (
        budget.config.GFSPHYSICS_VARIABLES
        + budget.config.PHYSICS_VARIABLES
        + budget.config.RESTART_VARIABLES
    ):
        assert name in dataset

    # 2d variable should stay 2d
    assert set(dataset["PRATEsfc_coarse"].dims) == set(
        ["step", "tile", "time", "grid_yt", "grid_xt"]
    )

    assert len(dataset["time"]) > 0
