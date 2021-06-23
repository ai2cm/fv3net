from typing import Mapping, List, no_type_check
import xarray as xr

from ._base import GeoMapper
from ._nudged import open_nudge_to_fine
from ._fine_resolution_budget import (
    FineResolutionSources,
    open_fine_res_apparent_sources,
)


class ResidualMapper(GeoMapper):
    def __init__(self, physics_mapper: GeoMapper, fine_res: FineResolutionSources):
        self.physics_mapper = physics_mapper
        self.fine_res = fine_res

    def keys(self):
        return list(set(self.physics_mapper.keys()) & set(self.fine_res.keys()))


class FineResolutionResidual(ResidualMapper):
    """Define the fine resolution dQ as a residual from the physics tendencies in
    another mapper
    """

    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        return nudging.assign(
            pQ1=nudging.pQ1,
            pQ2=nudging.pQ2,
            dQ1=fine_res.dQ1 - nudging.pQ1,
            dQ2=fine_res.dQ2 - nudging.pQ2,
        ).load()


T_Mapper = FineResolutionResidual


@no_type_check
def _open_fine_resolution_nudging_hybrid(
    data_paths: List[str],
    mapper: T_Mapper = FineResolutionResidual,
    nudging: Mapping = None,
    fine_res: Mapping = None,
    **kwargs
) -> T_Mapper:
    """
    Open the fine resolution nudging_hybrid mapper

    Args:
        data_paths: If list of urls is provided, the first is used as the nudging
            data url and second is used as fine res. If string or None, the paths must
            be provided in each mapper's kwargs.
        mapper: Hybrid mapper to use in opening the data
        nudging: keyword arguments passed to
            :py:func:`open_merged_nudging_full_tendencies`
        fine_res: keyword arguments passed to :py:func:`open_fine_res_apparent_sources`
        **kwargs: additional keyword arguments to be passed to the mapper constructor

    Returns:
        a mapper
    """
    nudging = nudging or {}
    fine_res = fine_res or {}
    offset_seconds = fine_res.pop("offset_seconds", 450)
    if isinstance(data_paths, (tuple, List)) and len(data_paths) == 2:
        nudging["url"] = data_paths[0]
        fine_res["fine_res_url"] = data_paths[1]
    else:
        # if not provided through data_paths, must be in kwargs dicts
        if "url" not in nudging or "fine_res_url" not in fine_res:
            raise ValueError(
                "Urls for nudging and fine res must be provided as either "
                "i) data_paths list arg of len 2, or ii) keys in nudging kwargs and "
                "fine res kwargs."
            )

    nudged = open_nudge_to_fine(**nudging)
    fine_res = open_fine_res_apparent_sources(offset_seconds=offset_seconds, **fine_res)
    return mapper(nudged, fine_res, **kwargs)


def open_fine_resolution_nudging_hybrid(
    data_paths: List[str], nudging: Mapping = None, fine_res: Mapping = None,
) -> FineResolutionResidual:
    """
    Open the fine resolution nudging_hybrid mapper

    Args:
        data_paths: If list of urls is provided, the first is used as the nudging
            data url and second is used as fine res. If string or None, the paths must
            be provided in each mapper's kwargs.
        nudging: keyword arguments passed to
            :py:func:`open_merged_nudging_full_tendencies`
        fine_res: keyword arguments passed to :py:func:`open_fine_res_apparent_sources`

    Returns:
        a mapper
    """
    return _open_fine_resolution_nudging_hybrid(
        data_paths, nudging=nudging, fine_res=fine_res
    )
