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
from typing import Mapping, Tuple

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
    #     print(sample_ds.data_vars)
    for var in TRAINING_REQUIRED_VARS:
        assert var in sample_ds.data_vars


REQUIRED_DIMENSIONS_2D_VARS = (
    ("tile",),
    ("y", "y_interface"),
    ("x", "x_interface"),
)


REQUIRED_DIMENSIONS_3D_VARS = (
    ("tile",),
    ("z",),
    ("y", "y_interface"),
    ("x", "x_interface"),
)


def _test_dimension_match(var_dims: Tuple[str], required_dims: Tuple[Tuple[str]]):
    if len(var_dims) != len(required_dims):
        return False
    for var_dim, required_dim in zip(var_dims, required_dims):
        if var_dim not in required_dim:
            return False
    return True


def test_training_mapper_dimensions(training_mapper):
    sample_ds = get_sample_dataset(training_mapper)
    for var in sample_ds.data_vars:
        #         print(sample_ds[var].dims)
        assert _test_dimension_match(
            sample_ds[var].dims, REQUIRED_DIMENSIONS_2D_VARS
        ) or _test_dimension_match(sample_ds[var].dims, REQUIRED_DIMENSIONS_3D_VARS)


def test_training_mapper_keys(training_mapper):
    keys = training_mapper.keys()
    assert isinstance(keys, set)
    for key in keys:
        assert isinstance(cast_to_datetime(key), datetime)
