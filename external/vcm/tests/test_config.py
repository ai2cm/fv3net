import pytest

from vcm import update_nested_dict


@pytest.mark.parametrize(
    "source, update, expected_output",
    [
        ({}, {}, {}),
        ({}, {"foo": "bar"}, {"foo": "bar"}),
        ({"foo": "bar"}, {}, {"foo": "bar"}),
        ({"foo": "bar"}, {"fee": "bee"}, {"foo": "bar", "fee": "bee"}),
        ({"foo": "bar"}, {"foo": "biz"}, {"foo": "biz"}),
        ({"foo": {"fee": "bar"}}, {"foo": {"fee": "biz"}}, {"foo": {"fee": "biz"}}),
        ({"foo": "bar", "fee": "bee"}, {"foo": "biz"}, {"foo": "biz", "fee": "bee"}),
    ],
)
def test_update_nested_dict(source, update, expected_output):
    output = update_nested_dict(source, update)
    assert expected_output == output
