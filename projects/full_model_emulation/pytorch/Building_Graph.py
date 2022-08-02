import torch
import numpy as np
import dgl
import geopy.distance
from vcm.catalog import catalog
import dataclasses


"""
Attributes:
    Nneighbor: number of nearest grid points
    selfpoint: Add self-loops for each node in the graph
    lat: latitude of cubed sphere data
    lon: longitude of the cubed sphere data
"""
@dataclasses.dataclass
class BuildingGraph:
    """
    Attributes:
        epochs: number of times to run through the batches when training
        shuffle_buffer_size: size of buffer to use when shuffling samples, only
            applies if in_memory=False
        coarsen: 1 if the full model resolution is used, othewise data will be coarsen
        Resolution: Model resolution to load the corresponding lat and lon.
    """

    Neighbor: int = 5
    coarsen: int = 1
    Resolution: str = "grid/c48"

def graphStruc(config):
    nodes=[]
    edges=[]

    grid = catalog[config.Resolution].read()
    lat=grid.lat.load()
    lon=grid.lon.load()

    lat=lat[:,::config.coarsen,::config.coarsen].values.flatten()
    lon=lon[:,::config.coarsen,::config.coarsen].values.flatten()
        
    lon=lon*180/np.pi
    lat=lat*180/np.pi
    for i in range(0,len(lat)):
        dis=np.zeros([len(lat)])
        coords_1 = (lat[i], lon[i])
        for j in range(0,len(lat)):
            coords_2 = (lat[j], lon[j])
            dis[j]=geopy.distance.geodesic(coords_1, coords_2).km
        res = sorted(range(len(dis)), key = lambda sub: dis[sub])[:config.Neighbor]
        nodes.append(np.repeat(i , config.Neighbor, axis=0))
        edges.append(res)
    nodes=torch.tensor(nodes).flatten()
    edges=torch.tensor(edges).flatten()
    with open('NodesEdges.npy', 'wb') as f:
        np.save(f, nodes)
        np.save(f, edges)
    g = dgl.graph((nodes,edges))
    return g