import pytest

from loaders import mappers


diagnostic_mapper_names = [
    "FineResolutionSources",
]


@pytest.fixture
def diagnostic_mapper(
    diagnostic_mapper_name,
    diagnostic_mapper_data_source_path,
    C48_SHiELD_diags_dataset_path,
):
    path = diagnostic_mapper_data_source_path
    if diagnostic_mapper_name == "FineResolutionSources":
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
    diagnostic_mapper_name, fine_res_dataset_path,
):
    if diagnostic_mapper_name == "FineResolutionSources":
        return fine_res_dataset_path
    else:
        return ValueError(f"{diagnostic_mapper_name} not implemented")
