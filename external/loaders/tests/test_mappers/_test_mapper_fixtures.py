from synth import data_source_name, data_source_path
import pytest
import xarray as xr
from loaders import mappers

@pytest.fixture
def training_mapper_helper_function(data_source_name):
    if data_source_name == "one_step_tendencies":
        return getattr(mappers, "open_one_step")
    elif data_source_name == "nudging_tendencies":
        return getattr(mappers, "open_merged_nudged")
    elif data_source_name == "fine_res_apparent_sources":
        # patch until synth is netcdf-compatible
        return None
    
    
@pytest.fixture
def training_mapper_helper_function_kwargs(data_source_name):
    if data_source_name == "one_step_tendencies":
        return {}
    elif data_source_name == "nudging_tendencies":
        return {'nudging_timescale_hr': 3}
    elif data_source_name == "fine_res_apparent_sources":
        # patch until synth is netcdf-compatible
        return None


@pytest.fixture
def training_mapper(
    data_source_name, data_source_path, training_mapper_helper_function, training_mapper_helper_function_kwargs
):
    if data_source_name != 'fine_res_apparent_sources':
        return training_mapper_helper_function(data_source_path, **training_mapper_helper_function_kwargs)
    else:
        # patch until synth is netcdf-compatible
        fine_res_ds = xr.open_zarr(data_source_path)
        mapper = {
            fine_res_ds.time.values[0]: fine_res_ds.isel(time=0),
            fine_res_ds.time.values[1]: fine_res_ds.isel(time=1),
        }
        return mapper