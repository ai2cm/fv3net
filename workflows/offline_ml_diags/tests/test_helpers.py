import pytest
from offline_ml_diags._helpers import sample_outside_train_range


@pytest.mark.parametrize(
    "train, all, n, expected_length, allowed_test_samples",
    [
        pytest.param(
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            2,
            2,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
            id="request_less_than_available",
        ),
        pytest.param(
            ["20160101.120000", "20160102.120000", "20160103.120000"],
            [
                "20160101.120000",
                "20160102.120000",
                "20160103.120000",
                "20150101.120000",
                "20160202.120000",
                "20160203.120000",
            ],
            10,
            3,
            ["20150101.120000", "20160202.120000", "20160203.120000"],
            id="request_more_than_available",
        ),
        pytest.param(
            [],
            ["20150101.120000", "20150102.120000"],
            1,
            1,
            ["20150101.120000", "20150102.120000"],
            id="no-config-set",
        ),
        pytest.param(
            ["20160101.120000", "20160102.120000"],
            ["20160101.120000", "20160102.120000"],
            3,
            None,
            None,
            id="error-none-avail-outside-config-range",
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
