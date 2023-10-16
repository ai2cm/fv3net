from typing import Any, Tuple
import pace.util
import numpy as np
import xarray as xr
from runtime.names import (
    TENDENCY_TO_STATE_NAME,
    A_GRID_WIND_TENDENCIES,
    D_GRID_WIND_TENDENCIES,
    EASTWARD_WIND_TENDENCY,
    NORTHWARD_WIND_TENDENCY,
    X_WIND_TENDENCY,
    Y_WIND_TENDENCY,
    STATE_NAME_TO_TENDENCY,
)
from runtime.types import State, Tendencies
from toolz import dissoc


def tendencies_from_state_updates(
    initial_state: State, updated_state: State, dt: float
) -> Tendencies:
    """Compute tendencies given intial and updated states

    Args:
        initial_state: initial state
        updated_state: updated state
        variables: variables to compute tendencies for

    Returns:
        tendencies: tendencies computed from state updates
    """
    tendencies = {}
    for variable in updated_state:
        tendency_var = STATE_NAME_TO_TENDENCY[variable]
        tendencies[tendency_var] = (
            updated_state[variable] - initial_state[variable]
        ) / dt
    return tendencies


def state_updates_from_tendency(tendency_updates):
    # Prescriber can overwrite the state updates predicted by ML tendencies
    # Sometimes this is desired and we want to save both the overwritten updated state
    # as well as the ML-predicted state that was overwritten, ex. reservoir updates.

    updates = {
        f"{k}_state_from_postphysics_tendency": v for k, v in tendency_updates.items()
    }

    return updates


def transform_from_agrid_to_dgrid(
    wrapper, u: xr.DataArray, v: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Transform a vector field on the A-grid in latitude-longitude coordinates
    to the D-grid in cubed-sphere coordinates.

    u and v must have double precision and contain units attributes.
    """
    u_quantity = pace.util.Quantity.from_data_array(u)
    v_quantity = pace.util.Quantity.from_data_array(v)
    (
        x_wind_quantity,
        y_wind_quantity,
    ) = wrapper.transform_agrid_winds_to_dgrid_winds(u_quantity, v_quantity)
    return x_wind_quantity.data_array, y_wind_quantity.data_array


def contains_agrid_tendencies(tendencies):
    return any(k in tendencies for k in A_GRID_WIND_TENDENCIES)


def contains_dgrid_tendencies(tendencies):
    return any(k in tendencies for k in D_GRID_WIND_TENDENCIES)


def fillna_tendency(tendency: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    tendency_filled = tendency.fillna(0.0)
    tendency_filled_frac = (
        xr.where(tendency != tendency_filled, 1, 0).sum("z") / tendency.sizes["z"]
    )
    tendency_filled_frac_name = f"{tendency_filled.name}_filled_frac"
    tendency_filled_frac = tendency_filled_frac.rename(tendency_filled_frac_name)
    return tendency_filled, tendency_filled_frac


def add_tendency(state: Any, tendencies: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state, which only includes
    variables updated by tendencies.  Tendencies cannot contain null values.
    """
    with xr.set_options(keep_attrs=True):
        updated: State = {}
        for name, tendency in tendencies.items():
            try:
                state_name = str(TENDENCY_TO_STATE_NAME[name])
            except KeyError:
                raise KeyError(
                    f"Tendency variable '{name}' does not have an entry mapping it "
                    "to a corresponding state variable to add to. "
                    "Existing tendencies with mappings to state are "
                    f"{list(TENDENCY_TO_STATE_NAME.keys())}"
                )

            updated[state_name] = state[state_name] + tendency * dt
    return updated


def fillna_tendencies(tendencies: State) -> Tuple[State, State]:
    filled_tendencies: State = {}
    filled_fractions: State = {}

    for name, tendency in tendencies.items():
        (
            filled_tendencies[name],
            filled_fractions[f"{name}_filled_frac"],
        ) = fillna_tendency(tendency)

    return filled_tendencies, filled_fractions


def prepare_agrid_wind_tendencies(
    tendencies: State,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Ensure A-grid wind tendencies are defined, have the proper units, and
    data type before being passed to the wrapper.

    Assumes that at least one of dQu or dQv appears in the tendencies input
    dictionary.
    """
    dQu = tendencies.get(EASTWARD_WIND_TENDENCY)
    dQv = tendencies.get(NORTHWARD_WIND_TENDENCY)

    if dQu is None:
        dQu = xr.zeros_like(dQv)
    if dQv is None:
        dQv = xr.zeros_like(dQu)

    dQu = dQu.assign_attrs(units="m/s/s").astype(np.float64, casting="same_kind")
    dQv = dQv.assign_attrs(units="m/s/s").astype(np.float64, casting="same_kind")
    return dQu, dQv


def transform_agrid_wind_tendencies(wrapper, tendencies: State) -> State:
    """Transforms available A-grid wind tendencies to the D-grid.

    Currently this does not support the case that both A-grid and D-grid
    tendencies are provided and will raise an error in that situation.  It would
    be straightforward to enable support of that, however.
    """
    if contains_dgrid_tendencies(tendencies):
        raise ValueError(
            "Simultaneously updating A-grid and D-grid winds is currently not "
            "supported."
        )

    dQu, dQv = prepare_agrid_wind_tendencies(tendencies)
    dQx_wind, dQy_wind = transform_from_agrid_to_dgrid(wrapper, dQu, dQv)
    tendencies[X_WIND_TENDENCY] = dQx_wind
    tendencies[Y_WIND_TENDENCY] = dQy_wind
    return dissoc(tendencies, *A_GRID_WIND_TENDENCIES)


def prepare_tendencies_for_dynamical_core(tendencies: State) -> Tuple[State, State]:
    # Filled fraction diagnostics are recorded on the original grid, since that
    # is where the na-filling occurs.
    filled_tendencies, tendencies_filled_frac = fillna_tendencies(tendencies)
    if contains_agrid_tendencies(filled_tendencies):
        filled_tendencies = transform_agrid_wind_tendencies(filled_tendencies)
    return filled_tendencies, tendencies_filled_frac
