from typing import Optional, Sequence, Union
import xarray as xr

from loaders.typing import Mapper
from ._nudged import open_nudge_to_fine
from ._fine_resolution_budget import open_fine_res_apparent_sources
from loaders._config import register_mapper_function


class PhysicsResidual(Mapper):
    """Replaces dQ1 and dQ2 in a base mapper with their difference from
    pQ1 and pQ2 values provided in a physics mapper
    """

    def __init__(self, physics_mapper: Mapper, base_mapper: Mapper):
        """
        Args:
            physics_mapper: Mapper whose datasets contain pQ1 and pQ2
            base_mapper: Mapper whose datasets contain dQ1 and dQ2
        """
        self.physics_mapper = physics_mapper
        self.base_mapper = base_mapper

    def __getitem__(self, key: str) -> xr.Dataset:
        physics = self.physics_mapper[key]
        base = self.base_mapper[key]

        return physics.assign(
            pQ1=physics.pQ1,
            pQ2=physics.pQ2,
            dQ1=base.dQ1 - physics.pQ1,
            dQ2=base.dQ2 - physics.pQ2,
        )

    def keys(self):
        return list(
            set(self.physics_mapper.keys()).intersection(self.fine_res_mapper.keys())
        )

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


@register_mapper_function
def open_fine_resolution_nudging_hybrid(
    data_path: str,
    fine_res_path: str,
    nudging_variables: Sequence[str],
    shield_diags_path: Optional[str] = None,
    physics_timestep_seconds: float = 900.0,
    consolidated_nudging_data: bool = True,
    fine_res_offset_seconds: Union[int, float] = 450,
) -> PhysicsResidual:
    """
    Open the fine resolution nudging_hybrid mapper

    Args:
        data_path: path to nudging data
        fine_res_path: path to fine res data
        shield_diags_path: path to directory containing a zarr store of SHiELD
            diagnostics coarsened to the nudged model resolution (optional)
        physics_timestep_seconds (float): physics timestep, i.e. dt_atmos
        consolidated_nudging_data (bool): whether zarrs at data_path have
            consolidated metadata
        fine_res_offset_seconds: amount to shift the keys forward by in seconds. For
            example, if the underlying data contains a value at the key
            "20160801.000730", a value off 450 will shift this forward 7:30
            minutes, so that this same value can be accessed with the key
            "20160801.001500"

    Returns:
        a mapper
    """
    nudged = open_nudge_to_fine(
        data_path,
        nudging_variables=nudging_variables,
        physics_timestep_seconds=physics_timestep_seconds,
        consolidated=consolidated_nudging_data,
    )
    fine_res = open_fine_res_apparent_sources(
        data_path=fine_res_path,
        shield_diags_path=shield_diags_path,
        offset_seconds=fine_res_offset_seconds,
    )
    return PhysicsResidual(nudged, fine_res)
