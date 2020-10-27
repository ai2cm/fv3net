import xarray as xr

from .coarsen import shift_edge_var_to_center

EAST_NORTH_WIND_TENDENCIES = ["dQu", "dQv"]
EDGE_TO_CENTER_DIMS = {"x_interface": "x", "y_interface": "y"}


def center_and_rotate_xy_winds(
    wind_rotation_matrix: xr.Dataset,
    x_component: xr.DataArray,
    y_component: xr.DataArray,
):
    """ Transform D grid x/y winds to A grid E/N winds.

    Args:
        wind_rotation_matrix : Dataset with rotation coefficients for
        x/y to E/N rotation. Can be found in catalog.
        x_component : D grid x wind
        y_component : D grid y wind
    """
    x_component = shift_edge_var_to_center(x_component, EDGE_TO_CENTER_DIMS)
    y_component = shift_edge_var_to_center(y_component, EDGE_TO_CENTER_DIMS)
    return eastnorth_wind_tendencies(wind_rotation_matrix, x_component, y_component)


def eastnorth_wind_tendencies(
    wind_rotation_matrix: xr.Dataset,
    x_component: xr.DataArray,
    y_component: xr.DataArray,
):
    eastward_tendency, northward_tendency = EAST_NORTH_WIND_TENDENCIES
    rotated = xr.Dataset()
    rotated[eastward_tendency] = (
        wind_rotation_matrix["eastward_wind_u_coeff"] * x_component
        + wind_rotation_matrix["eastward_wind_v_coeff"] * y_component
    )
    rotated[northward_tendency] = (
        wind_rotation_matrix["northward_wind_u_coeff"] * x_component
        + wind_rotation_matrix["northward_wind_v_coeff"] * y_component
    )
    return rotated
