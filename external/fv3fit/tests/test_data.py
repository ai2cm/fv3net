import pytest
from fv3fit._shared.data import check_validation_train_overlap


def test_check_validation_train_overlap():
    no_overlap = [[1, 2], [3, 4]]
    overlap = [[1, 2], [2, 4]]
    check_validation_train_overlap(*no_overlap)
    with pytest.raises(ValueError):
        check_validation_train_overlap(*overlap)
