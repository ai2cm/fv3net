import pytest
from loaders.mappers._base import GeoMapper, LongRunMapper
from loaders.mappers._fine_resolution_budget import (
    FineResolutionBudgetTiles,
    GroupByTime,
    FineResolutionSources,
)
from loaders.mappers._transformations import SubsetTimes
from loaders.mappers._nudged._legacy import (
    MergeNudged,
    NudgedStateCheckpoints,
    NudgedFullTendencies,
)
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
    MergeOverlappingData,
]


def _assert_unique(keys):
    assert len(set(keys)) == len(list(keys))


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
        assert _test_dimension_match(
            sample_ds[var].dims, REQUIRED_DIMENSIONS_2D_VARS
        ) or _test_dimension_match(sample_ds[var].dims, REQUIRED_DIMENSIONS_3D_VARS)


def test_training_mapper_keys(training_mapper):
    keys = training_mapper.keys()
    _assert_unique(keys)
    for key in keys:
        assert isinstance(cast_to_datetime(key), datetime)


DIAGNOSTIC_REQUIRED_VARS = [
    "pressure_thickness_of_atmospheric_layer",
    "air_temperature",
    "specific_humidity",
    "dQ1",
    "dQ2",
    "net_heating",
    "net_precipitation",
]

REQUIRED_DIMENSIONS_2D_PHYSICS_VARS = (
    ("derivation"),
    ("tile",),
    ("y", "y_interface"),
    ("x", "x_interface"),
)

PHYSICS_VARS = (
    "net_heating",
    "net_precipitation",
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_downward_shortwave_flux_at_surface",
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere",
    "total_sky_upward_shortwave_flux_at_surface",
    "total_sky_downward_longwave_flux_at_surface",
    "total_sky_upward_longwave_flux_at_top_of_atmosphere",
    "total_sky_upward_longwave_flux_at_surface",
    "sensible_heat_flux",
    "latent_heat_flux",
    "surface_precipitation_rate",
)


def test_diagnostic_mapper_variables(diagnostic_mapper):
    sample_ds = get_sample_dataset(diagnostic_mapper)
    for var in DIAGNOSTIC_REQUIRED_VARS:
        assert var in sample_ds.data_vars


def test_diagnostic_mapper_dimensions(diagnostic_mapper):
    sample_ds = get_sample_dataset(diagnostic_mapper)
    for var in sample_ds.data_vars:
        if var in PHYSICS_VARS:
            assert _test_dimension_match(
                sample_ds[var].dims, REQUIRED_DIMENSIONS_2D_PHYSICS_VARS
            )
        elif "z" not in sample_ds[var].dims:
            assert _test_dimension_match(
                sample_ds[var].dims, REQUIRED_DIMENSIONS_2D_VARS
            )
        else:
            assert _test_dimension_match(
                sample_ds[var].dims, REQUIRED_DIMENSIONS_3D_VARS
            )


def test_diagnostic_mapper_keys(diagnostic_mapper):
    keys = diagnostic_mapper.keys()
    _assert_unique(keys)
    for key in keys:
        assert isinstance(cast_to_datetime(key), datetime)
