import pytest
from loaders.mappers._base import GeoMapper, LongRunMapper
from loaders.mappers._fine_resolution_budget import (
    FineResolutionBudgetTiles,
    GroupByTime,
    FineResolutionSources,
)
from loaders.mappers._nudged import (
    MergeNudged,
    NudgedStateCheckpoints,
    SubsetTimes,
    NudgedFullTendencies,
)
from loaders.mappers._one_step import TimestepMapper
from loaders import mappers
from typing import Mapping

geo_mapper_subclasses = [
    GeoMapper,
    LongRunMapper,
    FineResolutionBudgetTiles,
    GroupByTime,
    FineResolutionSources,
    MergeNudged,
    NudgedStateCheckpoints,
    SubsetTimes,
    NudgedFullTendencies,
    TimestepMapper,
]


@pytest.fixture(params=geo_mapper_subclasses)
def geo_mapper_subclass(request):
    return request.param


def test_GeoMapper_subclass(geo_mapper_subclass):
    assert issubclass(geo_mapper_subclass, Mapping)
    assert callable(getattr(geo_mapper_subclass, "keys", None))


long_run_mapper_subclasses = [LongRunMapper, MergeNudged]


@pytest.fixture(params=long_run_mapper_subclasses)
def long_run_mapper_subclass(request):
    return request.param


def test_long_run_mapper_subclass(long_run_mapper_subclass):
    assert issubclass(long_run_mapper_subclass, LongRunMapper)


def test_training_mapper(training_mapper):
    print(training_mapper)
