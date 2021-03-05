from ._base import GeoMapper, XarrayMapper
from ._local import mapper_to_local

# mapping functions we mean to be able to specify in the
# model training configuration
from ._base import open_zarr
from ._fine_resolution_budget import open_fine_res_apparent_sources
from ._hybrid import (
    open_fine_resolution_nudging_hybrid,
    open_fine_resolution_nudging_hybrid_clouds_off,
    open_fine_resolution_nudging_to_obs_hybrid,
)
from ._nudged import (
    open_nudge_to_obs,
    open_nudge_to_fine,
    open_nudge_to_fine_multiple_datasets,
    open_merged_nudged,  # from here legacy mappers of nudged runs for compatibility
    open_merged_nudged_full_tendencies,
    open_merged_nudge_to_obs,
    open_merged_nudge_to_obs_full_tendencies,
    open_nudged_to_obs_prognostic,
    open_merged_nudged_full_tendencies_multiple_datasets,
)
