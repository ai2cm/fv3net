import numpy as np
from fv3fit.keras._models.shared.halos import append_halos
import xarray as xr
import dgl

def build_graph(nx_tile: int):
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

    g = dgl.graph((out[:, 0],out[:, 1]))

    return g