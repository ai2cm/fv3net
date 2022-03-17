import pytest
import xarray
from vcm import (
    precision,
    recall,
    true_positive_rate,
    false_positive_rate,
    accuracy,
    f1_score,
)


classification_score_funcs = [
    precision,
    recall,
    true_positive_rate,
    false_positive_rate,
    accuracy,
    f1_score,
]


@pytest.mark.parametrize("score", classification_score_funcs)
@pytest.mark.parametrize("dims", [["sample", "y"], ["sample"]])
def test_classification_score_func_different_means(score, dims):
    x = xarray.DataArray([[False, True]], dims=["sample", "y"])
    out = score(x, x, lambda x: x.mean(dims))
    assert set(x.dims) - set(dims) == set(out.dims)


@pytest.mark.parametrize("score", classification_score_funcs)
def test_classification_score_func_regtest(regtest, score):
    x = xarray.DataArray([False, True, False, True, True, False], dims=["sample"])
    y = xarray.DataArray([False, False, True, True, True, True], dims=["sample"])
    print(float(score(x, y)), file=regtest)


@pytest.mark.parametrize("score", classification_score_funcs)
def test_classification_score_error_if_not_binary(score):
    x = xarray.DataArray([1.0], dims=["sample"])
    with pytest.raises(ValueError):
        score(x, x)
