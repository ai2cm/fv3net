import pytest
import xarray as xr
from loaders.mappers import GeoMapper, MergeOverlappingData

OVERLAP_DIM = "derivation"

da = xr.DataArray(
    [0],
    dims=["x"],
    coords={"x": [0]},
)

da_overlap_dim = xr.DataArray(
    [[0]],
    dims=["x", OVERLAP_DIM],
    coords={"x": [0], OVERLAP_DIM: ["existing"]},
)


def dataset(variables):
    data_arrays = {}
    for var in variables:
        if "overlap" in var:
            data_arrays[var.strip("overlap").strip("_")] = da_overlap_dim
        else:
            data_arrays[var] = da
    return xr.Dataset(data_arrays)


class MockBaseMapper(GeoMapper):
    def __init__(self, ds):
        self._ds = ds
        self._keys = [f"2020050{i}.000000" for i in range(4)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._ds
        ds.coords["time"] = [key]
        return ds

    def keys(self):
        return self._keys


@pytest.fixture
def mock_merged_mapper(request):
    left_vars, right_vars = request.param
    left_mapper = MockBaseMapper(dataset(left_vars))
    right_mapper = MockBaseMapper(dataset(right_vars))
    return MergeOverlappingData([left_mapper, right_mapper], ["left", "right"]) 


@pytest.mark.parametrize(
    "mock_merged_mapper, overlapping_vars",
    (
        [[["var0"], ["var1"]], []],
        [[["var0", "var1"], ["var1"]], ["var1"]],
        [[["var0", "overlap_var1"], ["var1"]], ["var1"]],
    ), indirect=["mock_merged_mapper"])
def test_merging_mapper_overlap(mock_merged_mapper, overlapping_vars):
    test_key = list(mock_merged_mapper.keys())[0]
    for overlap_var in overlapping_vars:
        assert overlap_var in list(mock_merged_mapper[test_key].data_vars)
        assert OVERLAP_DIM in mock_merged_mapper[test_key][overlap_var].dims
