from radiation.preprocessing import _flatten, _merge
import pytest
import numpy as np

ARR = np.array(3)


@pytest.mark.parametrize(
    ["in_dict", "flattened_dict", "raises_type_error"],
    [
        pytest.param({"a": ARR}, {"a": ARR}, False, id="flat"),
        pytest.param({"a": {"b": ARR}}, {"a_b": ARR}, False, id="nested"),
        pytest.param({"a": 3}, None, True, id="type_error"),
    ],
)
def test__flatten(in_dict, flattened_dict, raises_type_error):
    if raises_type_error:
        with pytest.raises(TypeError):
            _flatten(in_dict)
    else:
        out = _flatten(in_dict)
        assert out == flattened_dict


@pytest.mark.parametrize(
    ["in_dicts", "merged_dict", "raises_value_error"],
    [
        pytest.param([{"a": 3}], {"a": 3}, False, id="single"),
        pytest.param([{"a": 3}, {"b": 3}], {"a": 3, "b": 3}, False, id="merge_two"),
        pytest.param([{"a": 3}, {"a": 3}], None, True, id="name_clash"),
    ],
)
def test__merge(in_dicts, merged_dict, raises_value_error):
    if raises_value_error:
        with pytest.raises(ValueError):
            _merge(*in_dicts)
    else:
        out = _merge(*in_dicts)
        assert out == merged_dict
