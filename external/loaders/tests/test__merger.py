import pytest
import xarray as xr
from loaders.mappers import GeoMapper, MergeOverlappingData

OVERLAP_DIM = "derivation"

da = xr.DataArray([0], dims=["x"], coords={"x": [0]},)

da_overlap_dim = xr.DataArray(
    [[0]], dims=["x", OVERLAP_DIM], coords={"x": [0], OVERLAP_DIM: ["existing"]},
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
    def __init__(self, ds, start_time=0, end_time=4):
        self._ds = ds
        self._keys = [f"2020050{i}.000000" for i in range(start_time, end_time)]

    def __getitem__(self, key: str) -> xr.Dataset:
        ds = self._ds
        ds.coords["time"] = [key]
        return ds

    def keys(self):
        return self._keys


@pytest.fixture
def mapper_different_vars(request):
    left_vars, right_vars = request.param
    left_mapper = MockBaseMapper(dataset(left_vars))
    right_mapper = MockBaseMapper(dataset(right_vars))
    left_name = "left" if "overlap" not in "".join(left_vars) else None
    right_name = "right" if "overlap" not in "".join(right_vars) else None
    return MergeOverlappingData(left_mapper, right_mapper, left_name, right_name)


@pytest.mark.parametrize(
    "mapper_different_vars, overlapping_vars, overlap_coords",
    (
        [[["var0"], ["var1"]], [], {}],
        [[["var0", "var1"], ["var1"]], ["var1"], {"left", "right"}],
        [[["var0", "overlap_var1"], ["var1"]], ["var1"], {"existing", "right"}],
    ),
    indirect=["mapper_different_vars"],
)
def test_merged_time_overlap(mapper_different_vars, overlapping_vars, overlap_coords):
    test_key = list(mapper_different_vars.keys())[0]
    test_ds = mapper_different_vars[test_key]
    for overlap_var in overlapping_vars:
        assert overlap_var in list(test_ds.data_vars)
        assert OVERLAP_DIM in test_ds[overlap_var].dims
        assert set(test_ds[OVERLAP_DIM].values) == overlap_coords


@pytest.fixture
def mock_mapper_different_times(request):
    left_start, left_end, right_start, right_end = request.param
    left_mapper = MockBaseMapper(dataset(["var0", "var1"]), left_start, left_end)
    right_mapper = MockBaseMapper(dataset(["var0", "var1"]), right_start, right_end)
    return MergeOverlappingData(left_mapper, right_mapper, "left", "right")


@pytest.mark.parametrize(
    "mock_mapper_different_times, overlapping_times",
    (
        [[0, 4, 0, 4], [f"2020050{i}.000000" for i in range(4)]],
        [[0, 4, 0, 3], [f"2020050{i}.000000" for i in range(3)]],
        [[0, 4, 5, 8], []],
    ),
    indirect=["mock_mapper_different_times"],
)
def test_merged_var_overlap(mock_mapper_different_times, overlapping_times):
    assert set(mock_mapper_different_times.keys()) == set(overlapping_times)


def test_fail_when_source_name_given_for_existing_dims():
    left_mapper = MockBaseMapper(dataset(["var0", "var1"]))
    right_mapper = MockBaseMapper(dataset(["overlap_var0", "overlap_var1"]))
    with pytest.raises(Exception):
        MergeOverlappingData(left_mapper, right_mapper, "left", "right")


def test__check_overlap_vars_dims():
    overlap_vars = ["var0", "var1"]
    invalid_dataset = xr.Dataset({"var0": da, "var1": da_overlap_dim})
    ds = dataset(overlap_vars)
    with pytest.raises(Exception):
        MergeOverlappingData._check_overlap_vars_dims(
            [invalid_dataset, ds], overlap_vars, overlap_dim=OVERLAP_DIM
        )
