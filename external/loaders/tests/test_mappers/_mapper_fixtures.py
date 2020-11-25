import pytest

from loaders import mappers

training_mapper_names = [
    "FineResolutionSources",
    "SubsetTimes",
]


@pytest.fixture(params=training_mapper_names)
def training_mapper_name(request):
    return request.param


@pytest.fixture
def training_mapper_data_source_path(
    training_mapper_name, nudging_dataset_path, fine_res_dataset_path,
):
    if training_mapper_name == "SubsetTimes":
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

    if training_mapper_name == "SubsetTimes":
        return mappers.open_merged_nudged_legacy(path)
    elif training_mapper_name == "NudgedFullTendencies":
        return mappers.open_merged_nudged_full_tendencies_legacy(
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
