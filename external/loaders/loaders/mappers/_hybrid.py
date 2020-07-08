from typing import Mapping

import xarray as xr

from ._base import GeoMapper
from ._nudged import open_merged_nudged_full_tendencies
from ._fine_resolution_budget import (
    FineResolutionSources,
    open_fine_res_apparent_sources,
)


class FineResolutionResidual(GeoMapper):
    """Define the fine resolution dQ as a residual from the physics tendencies in
    another mapper
    """

    def __init__(self, physics_mapper: GeoMapper, fine_res: FineResolutionSources):
        self.physics_mapper = physics_mapper
        self.fine_Res = fine_res

    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        return nudging.assign(
            pQ1=nudging.pQ1,
            pQ2=nudging.pQ2,
            dQ1=fine_res.dQ1 - nudging.pQ1,
            dQ2=fine_res.dQ2 - nudging.pQ2,
        ).load()


def open_fine_resolution_nudging_hybrid(
    _,  # need empty argument to work with current configuration system
    nudging_url: str,
    nudging_kwargs: Mapping,
    fine_res_url: str,
    fine_res_kwargs: Mapping,
) -> FineResolutionResidual:
    nudged = open_merged_nudged_full_tendencies(nudging_url, **nudging_kwargs)
    fine_res = open_fine_res_apparent_sources(fine_res_url, **fine_res_kwargs)
    return FineResolutionResidual(nudged, fine_res)
