from datetime import timedelta
import intake

from vcm.calc import apparent_source
from vcm.cubedsphere.coarsen import rename_centered_xy_coords, shift_edge_var_to_center

TIME_DIM = "initialization_time"
GRID_VARS = ["grid_lon", "grid_lat", "grid_lont", "grid_latt"]
INPUT_VARS = ["sphum", "T", "delp", "u", "v", "slmsk"]
TARGET_VARS = ["Q1", "Q2", "QU", "QV"]


def _filename_from_first_timestep(ds):
    timestep = min(ds[TIME_DIM].values).strftime("%Y%m%d.%H%M%S")
    return timestep + ".zarr"


def _round_time(t):
    if t.second == 0:
        return t.replace(microsecond=0)
    elif t.second == 59:
        return t.replace(microsecond=0) + timedelta(seconds=1)
    else:
        raise ValueError("Time value > 1 second from 1 minute timesteps.")


