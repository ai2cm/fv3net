from typing import Mapping, List, TypeVar
import xarray as xr

from ._base import GeoMapper
from ._nudged import open_merged_nudged_full_tendencies, open_nudged_to_obs_prognostic
from ._fine_resolution_budget import (
    FineResolutionSources,
    open_fine_res_apparent_sources,
)


CLOUDS_OFF_TEMP_TENDENCIES = [
    "tendency_of_air_temperature_due_to_longwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_shortwave_heating_assuming_clear_sky",
    "tendency_of_air_temperature_due_to_turbulence",
    "tendency_of_air_temperature_due_to_dissipation_of_gravity_waves",
]
CLOUDS_OFF_SPHUM_TENDENCIES = ["tendency_of_specific_humidity_due_to_turbulence"]


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


def _compute_clouds_off_pQ1(ds: xr.Dataset):
    """Compute the clouds off tendency of temperature assuming clouds off."""
    return sum([ds[variable] for variable in CLOUDS_OFF_TEMP_TENDENCIES])


def _compute_clouds_off_pQ2(ds: xr.Dataset):
    """Compute the clouds off tendency of specific humidity assuming clouds off."""
    return sum([ds[variable] for variable in CLOUDS_OFF_SPHUM_TENDENCIES])


class FineResolutionResidualCloudsOff(ResidualMapper):
    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        clouds_off_pQ1 = _compute_clouds_off_pQ1(nudging)
        clouds_off_pQ2 = _compute_clouds_off_pQ2(nudging)

        return nudging.assign(
            pQ1=clouds_off_pQ1,
            pQ2=clouds_off_pQ2,
            dQ1=fine_res.dQ1 - clouds_off_pQ1,
            dQ2=fine_res.dQ2 - clouds_off_pQ2,
        ).load()


class FineResolutionResidualCloudsOffWithNudging(ResidualMapper):
    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        clouds_off_pQ1 = _compute_clouds_off_pQ1(nudging)
        clouds_off_pQ2 = _compute_clouds_off_pQ2(nudging)

        return nudging.assign(
            pQ1=clouds_off_pQ1,
            pQ2=clouds_off_pQ2,
            dQ1=fine_res.dQ1 - clouds_off_pQ1 + nudging.dQ1,
            dQ2=fine_res.dQ2 - clouds_off_pQ2 + nudging.dQ2,
        ).load()


class FineResolutionResidualCloudsOffWithXSHiELDNudging(ResidualMapper):
    def __getitem__(self, key: str) -> xr.Dataset:
        nudging = self.physics_mapper[key]
        fine_res = self.fine_res[key]

        clouds_off_pQ1 = _compute_clouds_off_pQ1(nudging)
        clouds_off_pQ2 = _compute_clouds_off_pQ2(nudging)

        return nudging.assign(
            pQ1=clouds_off_pQ1,
            pQ2=clouds_off_pQ2,
            dQ1=fine_res.dQ1 - clouds_off_pQ1 + fine_res.air_temperature_nudging,
            dQ2=fine_res.dQ2 - clouds_off_pQ2,
        ).load()


T_Mapper = TypeVar(
    "T_Mapper",
    FineResolutionResidual,
    FineResolutionResidualCloudsOff,
    FineResolutionResidualCloudsOffWithNudging,
    FineResolutionResidualCloudsOffWithXSHiELDNudging,
)


def _open_fine_resolution_nudging_hybrid(
    data_paths: List[str],
    mapper: T_Mapper = FineResolutionResidual,
    nudging: Mapping = None,
    fine_res: Mapping = None,
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

    nudged = open_merged_nudged_full_tendencies(**nudging)
    fine_res = open_fine_res_apparent_sources(offset_seconds=offset_seconds, **fine_res)
    return mapper(nudged, fine_res)


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


def open_fine_resolution_nudging_hybrid_clouds_off(
    data_paths: List[str], nudging: Mapping = None, fine_res: Mapping = None,
) -> FineResolutionResidualCloudsOff:
    """
    Open the fine resolution nudging mapper with using clouds off physics tendencies.

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
        data_paths, FineResolutionResidualCloudsOff, nudging, fine_res
    )


def open_fine_resolution_nudging_hybrid_clouds_off_with_nudging(
    data_paths: List[str], nudging: Mapping = None, fine_res: Mapping = None,
) -> FineResolutionResidualCloudsOffWithNudging:
    """
    Open the fine resolution nudging mapper with using clouds off physics tendencies.

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
        data_paths, FineResolutionResidualCloudsOffWithNudging, nudging, fine_res
    )


def open_fine_resolution_nudging_hybrid_clouds_off_with_xshield_nudging(
    data_paths: List[str], nudging: Mapping = None, fine_res: Mapping = None,
) -> FineResolutionResidualCloudsOffWithXSHiELDNudging:
    """
    Open the fine resolution nudging mapper with using clouds off physics tendencies.

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
        data_paths, FineResolutionResidualCloudsOffWithXSHiELDNudging, nudging, fine_res
    )


def open_fine_resolution_nudging_to_obs_hybrid(
    data_paths, prog_nudge_kwargs: Mapping = None, fine_res_kwargs: Mapping = None,
) -> FineResolutionResidual:
    """
    Fine resolution nudging_hybrid mapper for merging with prognostic nudged to
    observations datasets. This differs from the existing fine res / nudging hybrid
    because the prognostic run data already has the tendency difference from the
    physics step saved.

    Args:
        data_paths: If list of urls is provided, the first is used as the nudging
            data url and second is used as fine res. If string or None, the paths must
            be provided in each mapper's kwargs.
        prog_nudge_kwargs: keyword arguments passed to
            :py:func:`open_nudged_to_obs_prognostic`
        fine_res_kwargs: keyword arguments passed to :py:func:
            `open_fine_res_apparent_sources`

    Returns:
        a mapper
    """
    prog_nudge_kwargs = prog_nudge_kwargs or {}
    fine_res_kwargs = fine_res_kwargs or {}

    offset_seconds = fine_res_kwargs.pop("offset_seconds", 450)
    if isinstance(data_paths, (tuple, List)) and len(data_paths) == 2:
        prog_nudge_kwargs["url"] = data_paths[0]
        fine_res_kwargs["fine_res_url"] = data_paths[1]
    else:
        # if not provided through data_paths, must be in kwargs dicts
        if "url" not in prog_nudge_kwargs or "fine_res_url" not in fine_res_kwargs:
            raise ValueError(
                "Urls for nudging and fine res must be provided as either "
                "i) data_paths list arg of len 2, or ii) keys in nudging kwargs and "
                "fine res kwargs."
            )

    # keep the nudging tendencies' original names (don't rename to dQ)
    if "rename_vars" not in prog_nudge_kwargs:
        prog_nudge_kwargs["rename_vars"] = {
            "tendency_of_air_temperature_due_to_fv3_physics": "pQ1",
            "tendency_of_specific_humidity_due_to_fv3_physics": "pQ2",
            "grid_xt": "x",
            "grid_yt": "y",
            "pfull": "z",
        }
    if "nudging_to_physics_tendency" not in prog_nudge_kwargs:
        prog_nudge_kwargs["nudging_to_physics_tendency"] = {
            "t_dt_nudge": "pQ1",
            "q_dt_nudge": "pQ2",
        }

    nudged_to_obs = open_nudged_to_obs_prognostic(**prog_nudge_kwargs)
    fine_res = open_fine_res_apparent_sources(
        offset_seconds=offset_seconds, **fine_res_kwargs
    )
    return FineResolutionResidual(nudged_to_obs, fine_res)
