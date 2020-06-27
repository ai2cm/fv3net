import pytest
import xarray as xr
from loaders import mappers

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
        return None


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
        # patch until synth is netcdf-compatible
        return None


@pytest.fixture
def training_mapper(
    training_mapper_name,
    training_mapper_data_source_path,
    training_mapper_helper_function,
    training_mapper_helper_function_kwargs,
):

    if training_mapper_name != "FineResolutionSources":
        return training_mapper_helper_function(
            training_mapper_data_source_path, **training_mapper_helper_function_kwargs
        )
    else:
        # patch until synth is netcdf-compatible
        fine_res_ds = xr.open_zarr(training_mapper_data_source_path)
        training_mapper = {
            fine_res_ds.time.values[0]: fine_res_ds.isel(time=0),
            fine_res_ds.time.values[1]: fine_res_ds.isel(time=1),
        }
        return training_mapper
