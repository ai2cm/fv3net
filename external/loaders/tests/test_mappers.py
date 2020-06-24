import pytest
from loaders.mappers._fine_resolution_budget import FineResolutionBudgetTiles, FineResolutionSources
from loaders.mappers._nudged import GeoMapper, NudgedTimestepMapper, MergeNudged, NudgedStateCheckpoints, NudgedFullTendencies
from loaders.mappers._one_step import TimestepMapper

base_mappers = [
    FineResolutionBudgetTiles,
    FineResolutionSources,
    GeoMapper,
    NudgedTimestepMapper,
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
    TimestepMapper
]

