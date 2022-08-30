from typing import Tuple
import numpy as np
from fv3fit.keras._models.shared.halos import append_halos
import xarray as xr
import functools
from ..system import DEVICE
import dgl
from .edge_connection import get_grid
import torch


def build_graph(nx_tile: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns two 1D arrays containing the start and end points of edges of the graph.

    Args:
        nx_tile: number of horizontal grid points on each tile of the cubed sphere
    """
    n_tile, nx, ny = 6, nx_tile, nx_tile
    n_points = n_tile * nx * ny
    ds = xr.Dataset(
        data_vars={
            "index": xr.DataArray(
                np.arange(n_points).reshape((n_tile, nx, ny, 1)),
                dims=["tile", "x", "y", "z"],
            )
        }
    )
    ds = append_halos(ds, n_halo=1).squeeze("z")
    index = ds["index"].values
    total_edges = n_points * 5
    out = np.empty((total_edges, 2), dtype=int)
    # left connections
    out[:n_points, 0] = index[:, 1:-1, 1:-1].flatten()
    out[:n_points, 1] = index[:, :-2, 1:-1].flatten()
    # right connections
    out[n_points : 2 * n_points, 0] = index[:, 1:-1, 1:-1].flatten()
    out[n_points : 2 * n_points, 1] = index[:, 2:, 1:-1].flatten()
    # up connections
    out[2 * n_points : 3 * n_points, 0] = index[:, 1:-1, 1:-1].flatten()
    out[2 * n_points : 3 * n_points, 1] = index[:, 1:-1, 2:].flatten()
    # down connections
    out[3 * n_points : 4 * n_points, 0] = index[:, 1:-1, 1:-1].flatten()
    out[3 * n_points : 4 * n_points, 1] = index[:, 1:-1, :-2].flatten()
    # self-connections
    out[4 * n_points : 5 * n_points, 0] = index[:, 1:-1, 1:-1].flatten()
    out[4 * n_points : 5 * n_points, 1] = index[:, 1:-1, 1:-1].flatten()

    return out[:, 0], out[:, 1]


@functools.lru_cache(maxsize=64)
def build_dgl_graph(nx_tile: int) -> dgl.DGLGraph:
    """
    Returns a DGL graph of the cubed sphere.
    """
    graph_data = build_graph(nx_tile)
    g = dgl.graph(graph_data)
    return g.to(DEVICE)


@functools.lru_cache(maxsize=64)
def build_dgl_graph_with_edge(nx_tile: int):
    """
    Returns
    g: A DGL graph of the cubed sphere.
    edge_relation: edge features contain lat and lon
                    relations between reference points and their neighbours
                    (use fro networks with eadge features)
    """
    graph_data = build_graph(nx_tile)
    graph_tensor = torch.tensor(graph_data)
    lon, lat = get_grid(nx_tile)
    lon = lon.flatten()
    lat = lat.flatten()
    lat_neighbour = lat[graph_tensor[1]]
    lon_neighbour = lon[graph_tensor[1]]
    lat_reference = lat[graph_tensor[0]]
    lon_reference = lon[graph_tensor[0]]
    lat_difference = lat_neighbour - lat_reference
    lon_difference = lon_neighbour - lon_reference
    edge_relation = [lat_difference.T, lon_difference.T]
    edge_relation = torch.from_numpy(np.swapaxes(edge_relation, 1, 0)).float()
    g = dgl.graph(graph_data)
    return g.to(DEVICE), edge_relation
