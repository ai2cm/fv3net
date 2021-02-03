import pytest
from offline_ml_diags._helpers import sample_outside_train_range


@pytest.mark.parametrize(
    "train, all, n, expected_length, allowed_test_samples",
    [
        (
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            3,
            3,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
        ),
        (
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            1,
            1,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
        ),
        ([], ["20150101.120000"], 3, None, None),
        (
            ["20160101.120000", "20160102.120000"],
            ["20160101.120000", "20160102.120000"],
            3,
            None,
            None,
        ),
    ],
)
def test_sample_outside_train_range(
    train, all, n, expected_length, allowed_test_samples
):
    if expected_length:
        test = sample_outside_train_range(all, train, n)
        assert len(test) == expected_length
        assert set(test).issubset(allowed_test_samples)
    else:
        with pytest.raises(ValueError):
            test = sample_outside_train_range(all, train, n)
