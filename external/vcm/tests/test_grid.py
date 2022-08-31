import numpy as np
import pytest
from vcm.grid import get_grid


@pytest.mark.parametrize("nx", [48, 24, 6])
def test_grid_builder(nx):
    lon, lat = get_grid(nx)
    assert np.all(np.diff(lat[1, 0, :]) > 0)
    assert np.all(np.diff(lon[1, :, 0]) > 0)
    assert (
        lon.shape[1] != nx
        and lon.shape[2] != nx
        and lat.shape[1] != nx
        and lat.shape[2] != nx
    )

    if len(lon.shape) != 3 and len(lat.shape) != 3:
        raise ValueError(
            "inputs must be of shape (n_tiles, n_x, n_y), "
            f"got lon {lon.shape} and lat {lat.shape}"
        )
