import torch
from scipy.spatial import KDTree
import numpy as np
import dgl

"""
Attributes:
    Nneighbor: number of nearest grid points
    selfpoint: Add self-loops for each node in the graph
    lat: latitude of cubed sphere data
    lon: longitude of the cubed sphere data
"""


class BuildingGraph:
    def __init__(self, Neighbor, lat, lon, selfpoint=True):
        super().__init__()
        self.Neighbor = Neighbor
        self.selfpoint = selfpoint
        self.lat = lat
        self.lon = lon

    def normalize_vector(self, *vector_components):
        scale = np.divide(
            1.0, np.sum(np.asarray([item ** 2.0 for item in vector_components])) ** 0.5
        )
        return np.asarray([item * scale for item in vector_components])

    def lon_lat_to_xyz(self):
        """map (lon, lat) to (x, y, z)
        Args:
            lon: 2d array of longitudes
            lat: 2d array of latitudes
            np: numpy-like module for arrays
        Returns:
            xyz: 3d array whose last dimension is length 3 and indicates x/y/z value
        """
        x = np.cos(self.lat) * np.cos(self.lon)
        y = np.cos(self.lat) * np.sin(self.lon)
        z = np.sin(self.lat)
        x, y, z = self.normalize_vector(x, y, z)
        if len(self.lon.shape) == 2:
            xyz = np.concatenate([arr[:, :, None] for arr in (x, y, z)], axis=-1)
        elif len(self.lon.shape) == 1:
            xyz = np.concatenate([arr[:, None] for arr in (x, y, z)], axis=-1)
        return xyz

    def graphStruc(self):
        xyz = self.lon_lat_to_xyz()

        kdtree = KDTree(xyz)

        _, points = kdtree.query(xyz, self.Neighbor + 1)

        if self.selfpoint:
            nodes = torch.tensor(
                np.repeat(points[:, 0], self.Neighbor + 1, axis=0)
                .reshape(-1, self.Neighbor + 1)
                .flatten()
            )
            edges = torch.tensor(points[:, 0 : self.Neighbor + 1]).flatten()
        else:
            nodes = torch.tensor(
                np.repeat(points[:, 0], self.Neighbor, axis=0)
                .reshape(-1, self.Neighbor)
                .flatten()
            )
            edges = torch.tensor(points[:, 1 : self.Neighbor + 1].flatten())

        g = dgl.graph((nodes, edges))
        return g
