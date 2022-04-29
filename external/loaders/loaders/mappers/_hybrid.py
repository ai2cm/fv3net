from typing import Sequence
import xarray as xr

from loaders.mappers._base import GeoMapper
from loaders.mappers._xarray import XarrayMapper
from loaders._config import mapper_functions
from loaders.mappers._fine_res import _open_merged_dataset, compute_budget, Approach
from loaders.mappers._fine_res_budget import FineResBudget


def compute_hybrid_budget(ds: xr.Dataset) -> xr.Dataset:
    ds["dQ1"] = ds.Q1 - ds.tendency_of_air_temperature_due_to_fv3_physics
    ds["dQ2"] = ds.Q2 - ds.tendency_of_specific_humidity_due_to_fv3_physics
    ds["dQxwind"] = ds.x_wind_tendency_due_to_nudging
    ds["dQywind"] = ds.y_wind_tendency_due_to_nudging
    return ds


@mapper_functions.register
def open_fine_resolution_nudging_hybrid(
    approach: str,
    fine_url: str,
    include_temperature_nudging: bool = False,
    additional_dataset_urls: Sequence[str] = None,
    use_fine_res_state: bool = True,
    use_fine_res_fluxes: bool = False,
) -> GeoMapper:
    """
    Open the fine-res mapper using several configuration options

    Args:
        approach: one of a set of available approaches.
        fine_url: url where coarsened fine resolution data is stored
        include_temperature_nudging: whether to include fine-res nudging in Q1
        additional_dataset_urls: sequence of urls to zarrs containing additional
            data to be merged into the resulting mapper dataset, e.g., ML input
            features, the dynamics nudging tendencies, and the dynamics differences
            as required by the above approaches
        use_fine_res_state: set standard name state variables to point to the fine-res
            data. Set to True if wanting to use fine-res state as ML inputs in training.
        use_fine_res_fluxes: set standard name surface and TOA flux diagnostic variables
            to point to the fine-res data. Set of True if wanting to use fine-res fluxes
            as ML inputs in training.

    Returns:
        a mapper
    """

    approach_enum = Approach[approach]

    merged: FineResBudget = _open_merged_dataset(
        fine_url=fine_url,
        additional_dataset_urls=additional_dataset_urls,
        use_fine_res_state=use_fine_res_state,
        use_fine_res_fluxes=use_fine_res_fluxes,
    )
    budget = compute_budget(merged, approach_enum, include_temperature_nudging)
    hybrid_budget = compute_hybrid_budget(budget)

    return XarrayMapper(hybrid_budget)
