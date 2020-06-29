import pytest
import xarray as xr
from loaders import mappers
from loaders.mappers._fine_resolution_budget import FineResolutionSources
from typing import Mapping, Sequence

training_mapper_names = [
    "FineResolutionSources",
    "SubsetTimes",
    "NudgedFullTendencies",
    "TimestepMapper",
]


@pytest.fixture(params=training_mapper_names)
def training_mapper_name(request):
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


def _open_fine_res_apparent_sources_patch(
    training_mapper_data_source_path: str,
    rename_vars: Mapping[str, str],
    drop_vars: Sequence[str],
    dim_order: Sequence[str],
):
    # this function is a patch for the actual one until synth is netcdf-compatible
    fine_res_ds = xr.open_zarr(training_mapper_data_source_path)
    time_mapper = {
        fine_res_ds.time.values[0]: (fine_res_ds.isel(time=0)),
        fine_res_ds.time.values[1]: (fine_res_ds.isel(time=1)),
    }
    return FineResolutionSources(
        time_mapper, rename_vars=rename_vars, drop_vars=drop_vars, dim_order=dim_order
    )


@pytest.fixture
def training_mapper_helper_function(training_mapper_name):
    if training_mapper_name == "TimestepMapper":
        return getattr(mappers, "open_one_step")
    elif training_mapper_name == "SubsetTimes":
        return getattr(mappers, "open_merged_nudged")
    elif training_mapper_name == "NudgedFullTendencies":
        return getattr(mappers, "open_merged_nudged_full_tendencies")
    elif training_mapper_name == "FineResolutionSources":
        # patch until synth is netcdf-compatible
        return _open_fine_res_apparent_sources_patch


@pytest.fixture
def training_mapper_helper_function_kwargs(training_mapper_name):
    if training_mapper_name == "TimestepMapper":
        return {}
    elif training_mapper_name == "SubsetTimes":
        return {"nudging_timescale_hr": 3}
    elif training_mapper_name == "NudgedFullTendencies":
        return {
            "nudging_timescale_hr": 3,
            "open_checkpoints_kwargs": {
                "checkpoint_files": ("after_dynamics.zarr", "after_physics.zarr")
            },
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
def training_mapper(
    training_mapper_data_source_path,
    training_mapper_helper_function,
    training_mapper_helper_function_kwargs,
):
    return training_mapper_helper_function(
        training_mapper_data_source_path, **training_mapper_helper_function_kwargs
    )
