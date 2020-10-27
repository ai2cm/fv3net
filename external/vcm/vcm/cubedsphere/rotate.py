from typing import Sequence
import xarray as xr

from .coarsen import shift_edge_var_to_center

EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
X_Y_WIND_TENDENCIES = ["dQxwind", "dQywind"]
EDGE_TO_CENTER_DIMS = {"x_interface": "x", "y_interface": "y"}


def _wind_rotation_needed(available_vars: Sequence[str]):
    # Returns False if existing wind vars are already in lat/lon components
    if set(EAST_NORTH_WIND_TENDENCIES).issubset(available_vars):
        return False
    elif set(X_Y_WIND_TENDENCIES).issubset(available_vars):
        return True
    else:
        raise KeyError(
            "If lat/lon winds are requested, dataset must have either i) "
            f"{EAST_NORTH_WIND_TENDENCIES} or ii) {X_Y_WIND_TENDENCIES} "
            "as data variables."
        )


def _center_d_grid_winds(ds: xr.Dataset):
    for edge_wind in X_Y_WIND_TENDENCIES:
        ds[edge_wind] = shift_edge_var_to_center(
            ds[edge_wind], EDGE_TO_CENTER_DIMS
        )
    return ds


def insert_eastnorth_wind_tendencies(wind_rotation_matrix: xr.Dataset, ds: xr.Dataset):
    if _wind_rotation_needed(ds.data_vars):
        x_tendency, y_tendency = X_Y_WIND_TENDENCIES
        ds = _center_d_grid_winds(ds)
        eastward_tendency, northward_tendency = EAST_NORTH_WIND_TENDENCIES
        ds[eastward_tendency] = (
            wind_rotation_matrix["eastward_wind_u_coeff"] * ds[x_tendency]
            + wind_rotation_matrix["eastward_wind_v_coeff"] * ds[y_tendency]
        )
        ds[northward_tendency] = (
            wind_rotation_matrix["northward_wind_u_coeff"] * ds[x_tendency]
            + wind_rotation_matrix["northward_wind_v_coeff"] * ds[y_tendency]
        )
    return ds