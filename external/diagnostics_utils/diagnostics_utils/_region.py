from typing import Tuple, Hashable
import xarray as xr
from dataclasses import dataclass


@dataclass
class RegionOfInterest:
    lat_bounds: Tuple[float]
    lon_bounds: Tuple[float]

    def average(self, dataset):
        return _average(dataset, self.lat_bounds, self.lon_bounds)


tropical_atlantic = RegionOfInterest(
    lat_bounds=[0, 15],
    lon_bounds=[-50, -20],
)

equatorial_zone = RegionOfInterest(
    lat_bounds=[-10, 10],
    lon_bounds=[0, 360]
)


def _average(
    dataset: xr.Dataset,
    lat_bounds: Tuple[float],
    lon_bounds: Tuple[float],
    dims: Tuple[Hashable] = None,
):
    """Average a dataset over a region of interest
    Args:
        dataset: the data to average, must contain, lat, lon, and area variables
        lat_bounds, lon_bounds: the bounds of the regional box
        dims: the spacial dimensions to average over.
    """

    if dims is None:
        dims = dataset["lat"].dims

    stacked = dataset.stack(space=dims)
    grid = stacked
    lon_bounds_pos = [lon + 360. if lon < 0 else lon for lon in lon_bounds]
    lat_mask = (grid.lat > lat_bounds[0]) & (grid.lat < lat_bounds[1])
    lon_mask = (grid.lon > lon_bounds_pos[0]) & (grid.lon < lon_bounds_pos[1])

    region = stacked.sel(space=lat_mask * lon_mask)
    out = (region * region.area).mean("space") / region.area.mean("space")

    for key in out:
        out[key].attrs.update(dataset[key].attrs)
    out.attrs.update(dataset.attrs)
    return out


