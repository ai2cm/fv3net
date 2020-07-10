import pytest

from loaders import mappers


diagnostic_mapper_names = [
    "FineResolutionSources",
    "NudgedFullTendencies",
    "TimestepMapperWithDiags",
]


@pytest.fixture
def diagnostic_mapper(
    diagnostic_mapper_name,
    diagnostic_mapper_data_source_path,
    C48_SHiELD_diags_dataset_path,
):
    path = diagnostic_mapper_data_source_path
    if diagnostic_mapper_name == "TimestepMapperWithDiags":
        return mappers.open_one_step(path, add_shield_diags=True)
    elif diagnostic_mapper_name == "NudgedFullTendencies":
        return mappers.open_merged_nudged_full_tendencies(
            path,
            shield_diags_url=C48_SHiELD_diags_dataset_path,
            open_checkpoints_kwargs=dict(
                checkpoint_files=("after_dynamics.zarr", "after_physics.zarr")
            ),
        )
    elif diagnostic_mapper_name == "FineResolutionSources":
        return mappers.open_fine_res_apparent_sources(
            path,
            shield_diags_url=C48_SHiELD_diags_dataset_path,
            rename_vars={
                "delp": "pressure_thickness_of_atmospheric_layer",
                "grid_xt": "x",
                "grid_yt": "y",
                "pfull": "z",
            },
            drop_vars=["time"],
            dim_order=("tile", "z", "y", "x"),
        )
    else:
        raise NotImplementedError()


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
