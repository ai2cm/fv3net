import pytest
from loaders.mappers._fine_resolution_budget import (
    FineResolutionBudgetTiles,
    FineResolutionSources,
)
from loaders.mappers._nudged import (
    GeoMapper,
    NudgedTimestepMapper,
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
)
from loaders.mappers._one_step import TimestepMapper
from vcm.convenience import parse_datetime_from_str

base_mappers = [
    FineResolutionBudgetTiles,
    FineResolutionSources,
    GeoMapper,
    NudgedTimestepMapper,
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
    TimestepMapper,
]


@pytest.fixture(params=base_mappers)
def base_mapper(request):
    return request.param


def test_base_mapper(base_mapper):
    assert len(base_mapper) == 2
    for key in base_mapper:
        assert parse_datetime_from_str(key)
