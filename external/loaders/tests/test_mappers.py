import pytest
from loaders.mappers._base import GeoMapper, LongRunMapper
from loaders.mappers._fine_resolution_budget import (
    FineResolutionBudgetTiles,
    FineResolutionSources,
)
from loaders.mappers._nudged import (
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
)
from loaders.mappers._one_step import TimestepMapper

# from vcm.convenience import parse_datetime_from_str
from collections.abc import Mapping

mapper_classes = [
    GeoMapper,
    LongRunMapper,
    FineResolutionBudgetTiles,
    FineResolutionSources,
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
    TimestepMapper,
]


@pytest.fixture(params=mapper_classes)
def mapper_class(request):
    return request.param


def test_base_mapper(mapper_class):
    assert issubclass(mapper_class, Mapping)
