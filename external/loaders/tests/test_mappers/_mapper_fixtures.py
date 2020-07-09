import pytest

from loaders import mappers

training_mapper_names = [
    "FineResolutionSources",
    "SubsetTimes",
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
        return one_step_dataset_path
    elif training_mapper_name == "SubsetTimes":
        return nudging_dataset_path
    elif training_mapper_name == "FineResolutionSources":
        return fine_res_dataset_path
    else:
        raise NotImplementedError(training_mapper_name)


@pytest.fixture
def training_mapper(
    training_mapper_name, training_mapper_data_source_path,
):
    path = training_mapper_data_source_path

    if training_mapper_name == "TimestepMapper":
        return mappers.open_one_step(path)
    elif training_mapper_name == "SubsetTimes":
        return mappers.open_merged_nudged(path)
    elif training_mapper_name == "NudgedFullTendencies":
        return mappers.open_merged_nudged_full_tendencies(
            path,
            open_checkpoints_kwargs={
                "checkpoint_files": ("after_dynamics.zarr", "after_physics.zarr")
            },
        )
    elif training_mapper_name == "FineResolutionSources":
        return mappers.open_fine_res_apparent_sources(
            path,
            rename_vars={
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            },
            drop_vars=["time"],
            dim_order=["tile", "z", "y", "x"],
        )


# TODO move this diagnostics code to a separate module
diagnostic_mapper_names = [
    "FineResolutionSources",
    "NudgedFullTendencies",
    "TimestepMapperWithDiags",
]


@pytest.fixture
def diagnostic_mapper(
    diagnostic_mapper_data_source_path,
    diagnostic_mapper_helper_function,
    diagnostic_mapper_helper_function_kwargs,
):
    return diagnostic_mapper_helper_function(
        diagnostic_mapper_data_source_path, **diagnostic_mapper_helper_function_kwargs
    )


@pytest.fixture(params=diagnostic_mapper_names)
def diagnostic_mapper_name(request):
    return request.param


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
def diagnostic_mapper_helper_function(diagnostic_mapper_name):
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        return mappers.open_one_step
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        return mappers.open_merged_nudged_full_tendencies
    elif diagnostic_mapper_name == "FineResolutionSources":
        return mappers.open_fine_res_apparent_sources


@pytest.fixture
def diagnostic_mapper_helper_function_kwargs(
    diagnostic_mapper_name, C48_SHiELD_diags_dataset_path
):
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        kwargs = {"add_shield_diags": True}
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        kwargs = {
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
