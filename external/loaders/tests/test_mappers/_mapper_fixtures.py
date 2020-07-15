import pytest

from loaders import mappers
from loaders.mappers._fine_resolution_budget import (
    FineResolutionSources,
    DERIVATION_FV3GFS_COORD,
)
from loaders.mappers._high_res_diags import open_high_res_diags
from loaders.mappers._merged import MergeOverlappingData
from loaders.constants import DERIVATION_SHiELD_COORD
from typing import Mapping, Sequence

training_mapper_names = ["FineResolutionSources", "SubsetTimes", "TimestepMapper"]

diagnostic_mapper_names = [
    "FineResolutionSources",
    "SubsetTimes",
    "TimestepMapper"
]


@pytest.fixture(params=training_mapper_names)
def training_mapper_name(request):
    return request.param


@pytest.fixture(params=diagnostic_mapper_names)
def diagnostic_mapper_name(request):
    return request.param


@pytest.fixture
def training_mapper_data_source_path(
    training_mapper_name,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
):
    if training_mapper_name == "TimestepMapper":
        training_mapper_data_source_path = one_step_dataset_path
    elif training_mapper_name == "SubsetTimes":
        return nudging_dataset_path
    elif training_mapper_name == "FineResolutionSources":
        training_mapper_data_source_path = fine_res_dataset_path
    return training_mapper_data_source_path


@pytest.fixture
def diagnostic_mapper_data_source_path(
    diagnostic_mapper_name,
    one_step_dataset_path,
    nudging_dataset_path,
    fine_res_dataset_path,
):
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        diagnostic_mapper_data_source_path = one_step_dataset_path
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        diagnostic_mapper_data_source_path = nudging_dataset_path
    elif diagnostic_mapper_name == "FineResolutionSources":
        diagnostic_mapper_data_source_path = fine_res_dataset_path
    return diagnostic_mapper_data_source_path


@pytest.fixture
def training_mapper(
    training_mapper_name, training_mapper_data_source_path,
):
    path = training_mapper_data_source_path

    if training_mapper_name == "TimestepMapper":
        return mappers.open_one_step(path)
    elif training_mapper_name == "SubsetTimes":
        return mappers.open_merged_nudged(path)
    elif training_mapper_name == "FineResolutionSources":
        return mappers.open_fine_res_apparent_sources(
            path,
            rename_vars={
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            }
        )
    return kwargs

@pytest.fixture
def diagnostic_mapper(
    diagnostic_mapper_name
    diagnostic_mapper_data_source_path,
    C48_SHiELD_diags_dataset_path
):
    path = diagnostic_mapper_data_source_path
    
    if diagnostic_mapper_name == 'TimestepMapperWithDiags':
        return mappers.open_one_step(
            path,
            add_shield_diags=True
        )
    elif diagnostic_mapper_name == 'NudgedFullTendencies':
        return mappers.open_merged_nudged_full_tendencies(
            path,
            shield_diags_url=C48_SHiELD_diags_dataset_path
        )
    elif diagnostic_mapper_name == 'FineResolutionSources':
        return mappers.open_fine_res_apparent_sources(
            path,
            shield_diags_url=C48_SHiELD_diags_dataset_path,
            rename_vars={
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            }
        )