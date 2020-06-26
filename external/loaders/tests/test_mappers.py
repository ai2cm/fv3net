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

# from vcm.convenience import parse_datetime_from_str
from collections.abc import Mapping

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


@pytest.fixture
def training_mapper_helper_function(data_source_name):
    if data_source_name == "one_step_tendencies":
        return getattr(mappers, "open_one_step")
    elif data_source_name == "nudging_tendencies":
        return getattr(mappers, "open_merged_nudged")
    elif data_source_name == "find_res_apparent_sources":
        return getattr(mappers, "open_fine_res_apparent_sources")


@pytest.fixture
def training_mapper(
    data_source_name, data_source_path, training_mapper_helper_function
):
    return training_mapper_helper_function(data_source_path)


def test_training_mapper(training_mapper):
    print(training_mapper)
