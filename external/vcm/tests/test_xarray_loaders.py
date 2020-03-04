import numpy as np
import xarray as xr

import vcm


def test_open_tiles(tmpdir):

    coords = {
        "tile": np.arange(6),
        "x": np.arange(48),
        "y": np.arange(48),
    }

    ds = xr.Dataset(
        {"a": (["tile", "y", "x"], np.random.sample((6, 48, 48)))}, coords=coords
    )

    for i in range(6):
        ds.isel(tile=i).to_netcdf(tmpdir.join(f"prefix.tile{i+1}.nc"))

    loaded = vcm.open_tiles(str(tmpdir.join("prefix")))

    xr.testing.assert_equal(ds, loaded)
