# flake8: noqa

from fv3post.fregrid import FregridLatLon
import xarray as xr
from vcm.catalog import catalog
import vcm
import numpy as np


def get_area(lat, lon, radius):
    dsinlat = np.diff(np.sin(np.deg2rad(lat)))
    dlon = np.diff(np.deg2rad(lon))
    area = dsinlat[:, None] * dlon[None, :] * radius ** 2
    return area


if __name__ == "__main__":
    fs = vcm.get_fs("gs://")
    ds_grid = (
        xr.concat(
            [
                xr.open_dataset(
                    fs.open(
                        f"gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C48/C48_grid.tile{i}.nc"
                    )
                )
                for i in range(1, 7)
            ],
            dim="tile",
        )
        .drop(["x", "y"])
        .rename_dims({"nx": "x", "ny": "y", "nxp": "x_interface", "nyp": "y_interface"})
    )
    ds_grid2 = (
        xr.concat(
            [
                xr.open_dataset(
                    fs.open(
                        f"gs://vcm-fv3config/data/grid_data/v1.0/C48/C48_grid.tile{i}.nc"
                    )
                )
                for i in range(1, 7)
            ],
            dim="tile",
        )
        .drop(["x", "y"])
        .rename_dims({"nx": "x", "ny": "y", "nxp": "x_interface", "nyp": "y_interface"})
    )
    print(ds_grid, ds_grid2)
    GRID = catalog["grid/c96"].read()
    print(
        "gs://vcm-ml-raw/2020-11-12-gridspec-orography-and-mosaic-data/C48/ total area: {}".format(
            ds_grid.area.sum().values
        )
    )
    print(
        "gs://vcm-fv3config/data/grid_data/v1.0/C48/ total area: {}".format(
            ds_grid2.area.sum().values
        )
    )
    print('catalog["grid/c96"] total area: {}'.format(GRID["area"].sum().values))
    radius = 6.371e6
    value = 1.0 / (6 * 96 * 96)
    value_per_unit_area = value / GRID["area"]
    ds = xr.Dataset(data_vars={"scalar": value_per_unit_area})
    print(
        "total globally integrated scalar before interpolation: {}".format(
            (ds.scalar * GRID["area"]).sum().values
        )
    )
    print(
        "total globally integrated scalar before interpolation using mosaic area: {}".format(
            (ds.scalar * ds_grid.area).sum().values
        )
    )
    fregridder = FregridLatLon("C96", 180, 360)
    ds_latlon = fregridder.regrid_scalar(ds)
    lat = np.arange(-90, 91, 1)
    lon = np.arange(0, 361, 1)
    area = get_area(lat, lon, radius)
    print(
        "total globaly integrated scalar after interpolation: {}".format(
            (ds_latlon.scalar.values * area).sum()
        )
    )
    ds["scalar_norm"] = ds["scalar"] / ds.scalar.sum()
    ds_latlon["scalar_norm"] = ds_latlon["scalar"] / ds_latlon.scalar.sum()
