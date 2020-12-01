# mapping functions we mean to be able to specify in the
# model training configuration
from ._fine_resolution_budget import open_fine_res_apparent_sources
from ._nudged import (
    open_nudge_to_obs,
    open_nudge_to_fine,
    open_nudge_to_fine_multiple_datasets,
    open_nudge_to_fine_rundir,
    open_merged_nudged,  # from here legacy mappers of nudged runs for compatibility
    open_merged_nudged_full_tendencies,
    open_merged_nudge_to_obs,
    open_merged_nudge_to_obs_full_tendencies,
    open_merged_nudged_full_tendencies_multiple_datasets,
)
from ._transformations import ValMap, KeyMap
from ._local import LocalMapper, mapper_to_local
from ._hybrid import (
    open_fine_resolution_nudging_hybrid,
    open_fine_resolution_nudging_hybrid_clouds_off,
    open_fine_resolution_nudging_to_obs_hybrid,
)

# additional open mapper functions
from ._high_res_diags import open_high_res_diags

# mapper classes used externally
from ._base import GeoMapper, LongRunMapper, MultiDatasetMapper, XarrayMapper
from ._merged import MergeOverlappingData
