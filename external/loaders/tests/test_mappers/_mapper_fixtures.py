import pytest
import xarray as xr
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
    "NudgedFullTendencies",
    "TimestepMapperWithDiags",
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
    elif training_mapper_name in ("SubsetTimes", "NudgedFullTendencies"):
        training_mapper_data_source_path = nudging_dataset_path
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


def _open_fine_res_apparent_sources_patch(
    training_mapper_data_source_path: str,
    rename_vars: Mapping[str, str] = None,
    drop_vars: Sequence[str] = None,
    dim_order: Sequence[str] = None,
    shield_diags_url: str = None,
):
    # this function is a patch for the actual one until synth is netcdf-compatible
    fine_res_ds = xr.open_zarr(training_mapper_data_source_path)
    time_mapper = {
        fine_res_ds.time.values[0]: (fine_res_ds.isel(time=0)),
        fine_res_ds.time.values[1]: (fine_res_ds.isel(time=1)),
    }

    fine_resolution_sources_mapper = FineResolutionSources(
        time_mapper, rename_vars=rename_vars, drop_vars=drop_vars, dim_order=dim_order
    )

    if shield_diags_url is not None:
        shield_diags_mapper = open_high_res_diags(shield_diags_url)
        fine_resolution_sources_mapper = MergeOverlappingData(
            shield_diags_mapper,
            fine_resolution_sources_mapper,
            source_name_left=DERIVATION_SHiELD_COORD,
            source_name_right=DERIVATION_FV3GFS_COORD,
        )

    return fine_resolution_sources_mapper


@pytest.fixture
def training_mapper_helper_function(training_mapper_name):
    if training_mapper_name == "TimestepMapper":
        return getattr(mappers, "open_one_step")
    elif training_mapper_name == "SubsetTimes":
        return getattr(mappers, "open_merged_nudged")
    elif training_mapper_name == "FineResolutionSources":
        # patch until synth is netcdf-compatible
        return _open_fine_res_apparent_sources_patch


@pytest.fixture
def diagnostic_mapper_helper_function(diagnostic_mapper_name):
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        return getattr(mappers, "open_one_step")
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        return getattr(mappers, "open_merged_nudged_full_tendencies")
    elif diagnostic_mapper_name == "FineResolutionSources":
        # patch until synth is netcdf-compatible
        return _open_fine_res_apparent_sources_patch


@pytest.fixture
def training_mapper_helper_function_kwargs(
    training_mapper_name, C48_SHiELD_diags_dataset_path
):
    if training_mapper_name == "TimestepMapper":
        return {'add_shield_diags': True}
    elif training_mapper_name == "SubsetTimes":
        return {}
    elif training_mapper_name == "NudgedFullTendencies":
        return {
            "shield_diags_url": C48_SHiELD_diags_dataset_path,
        }
    elif training_mapper_name == "FineResolutionSources":
        return {
            "rename_vars": {
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            },
            "drop_vars": ["time"],
            "dim_order": ("tile", "z", "y", "x"),
        }


@pytest.fixture
def diagnostic_mapper_helper_function_kwargs(
    diagnostic_mapper_name, C48_SHiELD_diags_dataset_path
):
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        kwargs = {}
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        kwargs = {
            "nudging_timescale_hr": 3,
            "shield_diags_url": C48_SHiELD_diags_dataset_path,
            "open_checkpoints_kwargs": {
                "checkpoint_files": ("after_dynamics.zarr", "after_physics.zarr")
            },
        }
    elif diagnostic_mapper_name == "FineResolutionSources":
        kwargs = {
            "shield_diags_url": C48_SHiELD_diags_dataset_path,
            "rename_vars": {
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            },
            "drop_vars": ["time"],
            "dim_order": ("tile", "z", "y", "x"),
        }
    return kwargs


@pytest.fixture
def training_mapper(
    training_mapper_data_source_path,
    training_mapper_helper_function,
    training_mapper_helper_function_kwargs,
):
    return training_mapper_helper_function(
        training_mapper_data_source_path, **training_mapper_helper_function_kwargs
    )


@pytest.fixture
def diagnostic_mapper(
    diagnostic_mapper_data_source_path,
    diagnostic_mapper_helper_function,
    diagnostic_mapper_helper_function_kwargs,
):
    return diagnostic_mapper_helper_function(
        diagnostic_mapper_data_source_path, **diagnostic_mapper_helper_function_kwargs
    )
