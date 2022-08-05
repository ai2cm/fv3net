import torch
import numpy as np
import geopy.distance
from vcm.catalog import catalog
import dataclasses


@dataclasses.dataclass
class GraphConfig:
    """
    Attributes:
        neighbor: number of nearest neighbor grids
        coarsen: 1 if the full model resolution is used, othewise data will be coarsen
        resolution: Model resolution to load the corresponding lat and lon.
    """

    neighbor: int = 10
    coarsen: int = 8
    resolution: str = "grid/c48"


def build_graph(config: GraphConfig) -> tuple:
    nodes = []
    edges = []

    grid = catalog[config.resolution].read()
    lat = grid.lat.load()
    lon = grid.lon.load()

    lat = lat[:, :: config.coarsen, :: config.coarsen].values.flatten()
    lon = lon[:, :: config.coarsen, :: config.coarsen].values.flatten()

    for i in range(0, len(lat)):
        distance = np.zeros([len(lat)])
        coords_1 = (lat[i], lon[i])
        for j in range(0, len(lat)):
            coords_2 = (lat[j], lon[j])
            distance[j] = geopy.distance.geodesic(coords_1, coords_2).km
        destination_grids = sorted(range(len(distance)), key=lambda i: distance[i])[
            : config.neighbor
        ]
        nodes.append(np.repeat(i, config.neighbor, axis=0))
        edges.append(destination_grids)
    nodes = torch.tensor(nodes).flatten()
    edges = torch.tensor(edges).flatten()
    return (nodes, edges)
