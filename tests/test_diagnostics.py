import pytest
import xarray as xr

from fv3net.diagnostics.data import merge_comparison_datasets


@pytest.fixture()
def dataset():
    return xr.Dataset(
        {"a": (["x"], [0, 1]), "b": (["x"], [0, 1]),}, coords={"x": [0, 1]}
    )


def test_merge_comparison_datasets_empty_in_dropped_variables(dataset):
    ds = dataset
    merged = merge_comparison_datasets(
        [ds, ds.drop("b")], ["all_vars", "just_a"], "run"
    )
    assert merged.sel(run="just_a").b.isnull().all().item()


def test_merge_comparison_with_singleton_coordinates(dataset):
    ds = dataset

    dss = [ds, ds.assign_coords(z=0)]
    merged = merge_comparison_datasets(dss, ["all_vars", "singleton-dim"], "run")
