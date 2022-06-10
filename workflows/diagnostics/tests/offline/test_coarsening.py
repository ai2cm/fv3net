from fv3net.diagnostics.offline._coarsening import res_from_string
import pytest


@pytest.mark.parametrize(
    ["string", "expected_res"],
    [
        pytest.param("c48", 48, id="c48"),
        pytest.param("c384", 384, id="c384"),
        pytest.param("c8", 8, id="c8"),
        pytest.param("c_something_invalid", "error", id="invalid_string_error"),
    ],
)
def test_res_from_string(string, expected_res):
    if expected_res != "error":
        res = res_from_string(string)
        assert res == expected_res
    else:
        with pytest.raises(ValueError, match=r"res_str must start with .*"):
            res_from_string(string)
