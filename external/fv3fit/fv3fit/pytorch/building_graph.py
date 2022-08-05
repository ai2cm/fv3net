import torch
import numpy as np
import dgl
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


def build_graph(config: GraphConfig) -> dgl.DGLGraph:
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
        distination_grids = sorted(range(len(distance)), key=lambda sub: distance[sub])[
            : config.neighbor
        ]
        nodes.append(np.repeat(i, config.neighbor, axis=0))
        edges.append(distination_grids)
    nodes = torch.tensor(nodes).flatten()
    edges = torch.tensor(edges).flatten()
    with open("NodesEdges.npy", "wb") as f:
        np.save(f, nodes)
        np.save(f, edges)
    g = dgl.graph((nodes, edges))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return g.to(device)
