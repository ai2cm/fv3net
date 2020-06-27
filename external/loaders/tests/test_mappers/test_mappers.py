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
from loaders.mappers._merged import MergeOverlappingData
from loaders._utils import get_sample_dataset
from vcm import cast_to_datetime
from datetime import datetime
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
    MergeOverlappingData,
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


TRAINING_REQUIRED_VARS = [
    "pressure_thickness_of_atmospheric_layer",
    "air_temperature",
    "specific_humidity",
    "dQ1",
    "dQ2",
]


def test_training_mapper_variables(training_mapper):
    sample_ds = get_sample_dataset(training_mapper)
    for var in TRAINING_REQUIRED_VARS:
        assert var in sample_ds.data_vars


REQUIRED_DIMENSIONS = [
    ("tile"),
    ("z"),
    ("y", "y_interface"),
    ("x", "x_interface"),
]


def test_training_mapper_dimensions(training_mapper):
    sample_ds = get_sample_dataset(training_mapper)
    for var in sample_ds.data_vars:
        assert len(sample_ds[var].dims) == len(REQUIRED_DIMENSIONS)
        for mapper_dim, required_dim in zip(sample_ds[var].dims, REQUIRED_DIMENSIONS):
            assert mapper_dim in required_dim


def test_training_mapper_keys(training_mapper):
    keys = training_mapper.keys()
    assert isinstance(keys, set)
    for key in keys:
        assert isinstance(cast_to_datetime(key), datetime)
